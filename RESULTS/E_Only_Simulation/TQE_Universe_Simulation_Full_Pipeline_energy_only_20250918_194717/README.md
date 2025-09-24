SPDX-License-Identifier: MIT

Copyright (c) 2025 Stefan Len

[![CI](https://github.com/SteviLen420/TQE_simulation/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/SteviLen420/TQE_simulation/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.9%20|%203.10%20|%203.11-blue)](https://www.python.org/doc/)

# TQE Simulation Results for E-Only Universe (Run ID: 20250918_194717)
### Author: Stefan Len

This document presents a comprehensive analysis of a numerical simulation designed to test the **Theory of the Question of Existence (TQE)** under a control condition. In this "Energy-Only" (E-Only) variant, 10,000 universes were simulated where the posited **Information (I) parameter was set to zero**. This allows for the isolation of vacuum energy's role in the cosmogenesis process, providing a baseline against which the full E+I model can be compared. The analysis integrates results from the simulation's parameter logs, summary statistics, and the complete time-series data for all simulated universes.

---

## 1. The Initial Phases of Cosmogenesis: From Fluctuation to Collapse

The TQE model posits that a universe originates from a quantum fluctuation in a pre-law vacuum. Our simulation models this process through three distinct phases: **quantum fluctuation**, **superposition**, and **state reduction (collapse)**.

### The Fluctuation and Superposition Phase
The initial state is modeled as a normalized quantum fluctuation. Statistical analysis of the simulation's fluctuation phase confirms this setup: the expectation value of the system's state variable, $$A$$, remains centered around zero ($$\mathbb{E}[A] \approx -0.0167$$) with a variance approaching unity ($$\text{Var}(A) \approx 0.996$$). This establishes a statistically unbiased "blank slate" from which cosmic evolution can proceed.

Following this, the system enters a superpositional phase, exploring a vast space of potential outcomes. This is quantified by the system's **Shannon entropy ($$H$$)**, which reaches a mean value of **0.959** (on a normalized scale of 0 to 1). This near-maximal entropy indicates that, prior to collapse, the proto-universe exists as a rich superposition of possibilities, consistent with the theoretical framework of a system not yet constrained by deterministic laws.

### The Collapse Phase and the Primacy of Energy
The transition from a multi-potential state to a singular reality occurs during the collapse phase. At this critical juncture, the universe's defining characteristic, its complexity parameter $$X$$, is fixed. In this E-Only control run, the Information parameter $$I$$ is axiomatically set to zero ($$I=0$$). Consequently, the complexity parameter $$X$$ becomes directly and exclusively dependent on the initial vacuum energy fluctuation $$E$$:

$$
X = E
$$

This is a crucial feature of the control model. Unlike the full TQE framework, where complexity arises from a coupling of energy and information, here the potential for structure is entirely determined by the energetic magnitude of the initial fluctuation. The simulation confirms this, with the value of $$X$$ perfectly correlating with $$E$$ across all 10,000 universes.

---

## 2. Stability and the Energetic "Goldilocks" Window

A central prediction of TQE is that stable, law-abiding universes can only form within a specific energetic range, or "Goldilocks Zone." The E-Only simulation was designed to test whether such a zone emerges from purely energetic considerations, without the influence of the Information parameter.

### Emergence of a Stability Zone
The simulation results demonstrate a clear, probabilistic dependence of stability on energy. Out of 10,000 universes, **5,362 (53.6%) achieved a stable state**, and of those, **2,476 (24.8% of total) successfully reached the "lock-in" criterion**, defined as a relative parameter variation below 0.5% for at least 6 consecutive epochs.

The fine-tuning function, which biases the probability of state reduction towards stable outcomes, simplifies in the E-Only model to:

$$
P'(\psi) = P(\psi) \cdot f(E)
$$

where $$f(E)$$ is a function dependent only on energy. While the theoretical model suggests a Gaussian form centered at $$E_c = 4.0$$, the simulation employed a **dynamic Goldilocks window** that adapted to the lognormal distribution of initial energy values. This resulted in an effective stability window where universes with an energy $$E$$ (and thus $$X$$) between **5.06 and 93.47** had a non-zero probability of stabilizing.

A machine learning classifier trained solely on energy-derived features to predict stability achieved an **Area Under the Curve (AUC) of 0.618**. An AUC of 0.5 represents random chance, while 1.0 represents perfect prediction. This result confirms two key insights:
1.  Energy is a **statistically significant predictor** of stability.
2.  The relationship is **probabilistic, not deterministic**. Being within the Goldilocks window increases the chance of stabilization but does not guarantee it.

This supports the TQE hypothesis that an energetic Goldilocks zone is a necessary condition for law formation. However, in this E-Only scenario, the **two-level selection principle** (where $$X$$ is selected for stability and $$|E-I|$$ is selected for persistence) is absent. Stability is governed by a single, purely energetic probabilistic filter.

---

## 3. The Archetype of an Energy-Only Universe

By examining the universes that successfully achieved lock-in, we can characterize the archetype of a stable, Energy-Only cosmos. Their primary shared feature is a distinct and irreversible evolutionary trajectory, establishing a clear **arrow of time**.

### Entropic Evolution and Expansion

The post-collapse expansion phase, tracked over 800 epochs, shows a consistent, near-monotonic increase in the cosmic scale factor, $$A$$, with its mean value reaching **174**. This expansionary dynamic, emergent from the model's rules, serves as a proxy for the arrow of time. The simulation data suggests that once an Energy-Only universe locks into a stable state, its evolution follows a predictable, expansion-driven path.

These results lead to the formulation of a key hypothesis: _"In Energy-Only universes, stabilization may lead to simpler and more rigid worlds than in models that also include an Energy-Information coupling (E+I)."_

The current Energy-Only simulation provides a theoretical basis for this proposition. With the Information parameter $$I=0$$, the lock-in dynamics are simplified; the system does not need to negotiate a balance between energy and information. Consequently, the resulting stable universes can be considered more "rigid," as their physical laws are a direct consequence of the initial energy and the static Goldilocks filter. A co-evolving informational parameter, which could introduce more complex or adaptive stabilization pathways, is absent.

It is important to emphasize, however, that this is a theoretical inference that **is not validated by the current data alone**, as a comparative 'E+I' run is missing. The present findings, therefore, do not prove this hypothesis but rather frame it for future research. This sets the stage for testing whether the introduction of the information parameter would indeed lead to the formation of richer and more complex structures.

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
