SPDX-License-Identifier: MIT

Copyright (c) 2025 Stefan Len

[![CI](https://github.com/SteviLen420/TQE_simulation/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/SteviLen420/TQE_simulation/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.9%20|%203.10%20|%203.11-blue)](https://www.python.org/doc/)

# TQE Simulation Results for E-Only Universe (Run ID: 20250918_203906)
### Author: Stefan Len


This document presents a comprehensive analysis of the numerical simulation for the Theory of the Question of Existence (TQE), specifically focusing on the **E-Only control variant**. In this configuration, the information orientation parameter ($I$) is set to zero, isolating the role of vacuum energy fluctuation ($E$) in the cosmogenesis and stabilization of physical law. The simulation encompassed 10,000 universes, each evolving over 3,000 time steps.

---

### 1. The Initial Phases of Cosmogenesis: From Fluctuation to Collapse

The TQE model posits that a universe originates from a quantum fluctuation, transitioning through distinct phases of superposition and collapse to establish a foundational physical state. The E-Only simulation successfully replicates this process, providing a baseline against which the influence of information can be measured in future studies.

#### 1.1. Quantum Fluctuation and Superposition
The initial phase, represented by the `fl_fluctuation_timeseries` data, models the emergence of a proto-state from the vacuum. The simulation confirms that this phase establishes a statistically normalized "blank slate," characterized by a near-zero expectation value (`exp_A` mean: $1.57 \times 10^{-3}$) and unit variance (`var_A` mean: $0.9998$). This ensures that no pre-existing structure biases the subsequent evolution.

Following this, the system enters a superposition state, where all potential outcomes coexist. The `fl_superposition_timeseries` data indicates that this phase rapidly evolves toward maximum informational entropy (mean entropy: **0.969** on a normalized scale) and correspondingly low purity (mean purity: **0.277**). This represents the system exploring its entire possibility space before a definitive state is actualized, consistent with the theoretical premise of an unbiased emergence from a pre-law condition.

#### 1.2. State Collapse and Complexity Parameter Fixation
The superposition phase concludes with a quantum state collapse, a critical event where a single reality is actualized. In the TQE framework, this event fixes the universe's primary complexity parameter, $X$. For the E-Only model, this relationship is direct and unambiguous:

$$ X = E $$

The `fl_collapse_timeseries` data illustrates this moment, showing the parameter $X$ stabilizing from a fluctuating state to a locked-in value, $X_{\text{lock}}$ (mean $X$: 12.22 vs. mean $X_{\text{lock}}$: 12.18). This differs fundamentally from the proposed E+I model, where complexity would arise from the interplay between energy and information ($X = f(E, I)$). In this control run, the capacity for future complexity is determined entirely by the initial energy of the vacuum fluctuation.

---

### 2. Stability and the Emergence of a "Goldilocks" Energy Window

A central prediction of the TQE is that stable physical laws can only emerge within specific energetic boundaries. The simulation results for the E-Only case robustly demonstrate this phenomenon, revealing a clear, non-monotonic relationship between energy and stability.

#### 2.1. The Goldilocks Window for Stability
Out of **10,000** simulated universes, **5,080 (50.8%)** achieved stability, with **2,219 (22.2%)** subsequently achieving "lock-in" of their physical parameters according to the model's criteria ($ \Delta P / P < 0.005 $ over 6 epochs).

Analysis of the `tqe_runs` data, visualized in the `stability_distribution` plot, reveals that stability is not a simple monotonic function of energy. Instead, stable universes are predominantly found within a specific energetic range, or **"Goldilocks window"**. The simulation's dynamic fine-tuning function operated within a window where the complexity parameter $X$ (here, equivalent to $E$) was between **5.90** and **111.90**. While universes with very low or extremely high energy are overwhelmingly unstable, the probability of stabilization rises sharply, peaks, and then gradually declines for higher energy values. This suggests that a certain minimum energy is required to initiate stabilization, but excessively high energy introduces chaotic dynamics that are equally detrimental to the formation of stable laws.

#### 2.2. The Simplified Selection Principle
In the full TQE model, a two-level selection principle is hypothesized: the complexity parameter $X$ must fall within the Goldilocks window for **stability**, while the absolute difference $|E-I|$ must be minimized for **lock-in** (the permanent crystallization of laws).

In this E-Only simulation where $I=0$ by definition, this two-level principle collapses into a single, energy-dependent mechanism. Both stability and subsequent lock-in are governed by $E$ alone. The probability update rule simplifies from $ P'(\psi) = P(\psi) \cdot f(E,I) $ to:

$$ P'(\psi) = P(\psi) \cdot f(E) $$

The lock-in phase, therefore, does not depend on aligning energy and information but becomes a probabilistic outcome for universes that have already achieved stability within the correct energy window. This control simulation thus establishes a crucial baseline: **energy alone is sufficient to create a Goldilocks stability zone, but the subsequent lock-in of physical laws remains a probabilistic, not a deterministic, outcome.**

---

### 3. Archetype of a Successful Universe in an E-Only Cosmos

The E-Only simulation provides a clear profile of what constitutes a "successful" (i.e., stable and law-abiding) universe when information orientation is absent. The data suggests that such universes are not exceptional outliers but rather represent a significant, well-defined sub-population.

The `metrics_joined` dataset reveals that the mean energy ($E$) for the entire population of 10,000 universes is **16.90**. However, a detailed analysis of the stable and locked-in sub-populations would be required to determine if they favor a specific part of the energy distribution within the Goldilocks window. For instance, do universes that lock-in tend to have energies closer to the critical energy center ($E_c=4.0$)?

Without comparative data from an E+I simulation, we cannot confirm the TQE hypothesis that informational orientation acts as a "finishing kick" to guarantee lock-in for stable universes. However, the results from this E-Only run **provide the necessary theoretical groundwork for such a future test**. The finding that 50.8% of universes stabilize but only 22.2% achieve lock-in **raises the hypothesis** that an additional mechanism, such as the proposed information parameter $I$, may be responsible for converting transient stability into permanent, complexity-permitting law.

---

### 4. Testable Predictions: Cosmological Anomalies as Stabilization Relics

A key feature of the TQE framework is its potential to make falsifiable predictions concerning large-scale cosmological observations. The model suggests that the process of law stabilization may leave statistical imprints on the Cosmic Microwave Background (CMB), manifesting as the observed anomalies.

#### 4.1. Rarity and Nature of Anomalies
The simulation results confirm the emergence of features analogous to the **"Axis of Evil" (AoE)** and the **Cold Spot** in a small fraction of universes. The `metrics_joined` data shows that both `aoe_flag` and `cold_flag` have a mean occurrence rate of **0.0003**, corresponding to just **3 universes out of 10,000** for each anomaly. This extreme rarity is consistent with the idea that these are residual signatures of a highly specific cosmogenesis process.

#### 4.2. Correlation with "Best Case" Universes
An essential question is whether these anomalies are random occurrences or are correlated with the "best case" (i.e., most successfully stabilized) universes. The `cmb_aoe_summary` and `cmb_coldspots_summary` data provide initial insights. The six universes exhibiting these anomalies are all confirmed to be **locked-in universes**, achieving lock-in at epoch 305. This finding, though based on a small sample size, strongly suggests that the mechanisms producing CMB anomalies in this model are intrinsically linked to the process of successful law stabilization. It refutes the null hypothesis that such features are random artifacts and positions them as potential observational probes into the fundamental dynamics of cosmogenesis as modeled by TQE.


---

### Methodological Note on the Analytical Workflow

The primary conclusions presented in this document are derived from the direct statistical analysis of the simulation’s output data. This includes the aggregate statistics from the `summary_full.json`, the per-file checks from the `Wolfram math_check.json`, as well as the detailed run-level data contained in `tqe_runs_E-Only.csv`. This direct approach yielded robust and consistent findings.

In addition to the direct analysis, an extensive suite of predictive machine learning models was developed to probe the system's dynamics (the XAI module). While this framework is functional, the models themselves exhibited significant limitations, including overfitting and internally inconsistent explanations (as shown by conflicting SHAP and LIME results). For this reason, the conclusions from these predictive models are considered preliminary and have been excluded from the main findings of this initial publication, representing an area for future research.

The interpretation and articulation of the final analysis were performed in collaboration with several Large Language Models. An initial analysis attempt using a local DeepSeek model resulted in significant quantitative hallucinations. The final, validated analyses presented here were therefore developed and cross-checked through an iterative, Socratic dialogue with **Google's Gemini 2.5 Pro** and **OpenAI's GPT-5**, which acted as scientific reasoning and writing assistants based on the verified, factual data from the simulation.

----

## License
This project is licensed under the MIT License – see the [LICENSE](../../LICENSE) file for details.

## Contact

Got questions, ideas, or feedback?  
Drop me an email at **tqe.simulation@gmail.com** 
    
[RESULTS](../../RESULTS)
