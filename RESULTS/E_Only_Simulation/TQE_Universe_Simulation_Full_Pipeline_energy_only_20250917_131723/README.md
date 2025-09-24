SPDX-License-Identifier: MIT

Copyright (c) 2025 Stefan Len

[![CI](https://github.com/SteviLen420/TQE_simulation/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/SteviLen420/TQE_simulation/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.9%20|%203.10%20|%203.11-blue)](https://www.python.org/doc/)

# TQE Simulation Results for E-Only Universe (Run ID: 20250917_131723)
### Author: Stefan Len


This document presents a comprehensive analysis of a numerical control simulation conducted under the **Theory of the Question of Existence (TQE)** framework. The simulation models a cosmogenesis scenario where the **Information Orientation parameter ($I$) is set to zero**, isolating the role of vacuum energy ($E$) as the sole driver of complexity and stability. By examining 10,000 simulated universes, we investigate the initial phases of state evolution, the mechanisms of physical law stabilization, the characteristics of successful universes, and the emergence of cosmological anomalies. The results demonstrate that while energy alone can produce stable universes, it does so within a defined energetic "Goldilocks" window and fails to systematically generate the large-scale coherent structures predicted by the full E+I model, thus reinforcing the hypothesized role of information as a critical component for cosmic complexity.

***

## 1. Initial Phases of Cosmogenesis: From Quantum Fluctuation to a Defined State

The TQE model posits that cosmogenesis initiates from a quantum fluctuation, evolving through distinct phases of superposition, collapse, and expansion. In this E-Only control simulation, these initial steps establish the foundational state of each universe before the dynamics of stabilization are tested.

### 1.1 Fluctuation, Superposition, and Collapse

The simulation begins by modeling the primordial quantum state. The **fluctuation phase** represents the initial energy perturbation, followed by a **superposition phase** where the universe exists as a space of potentialities. As indicated by the statistical summary of `fl_superposition_timeseries_E-Only.csv`, this phase is characterized by a rapid increase in **Shannon entropy** towards its maximum value (mean $H \approx 0.96$). This demonstrates that the model successfully generates a state of maximal indeterminacy, a "blank slate" where no specific outcome is yet favored.

This state of maximal entropy is resolved during the **collapse phase**. The simulation models a quantum state reduction where the superposition collapses into a single, definite physical state. The core dynamic of this control run is defined at this moment. The complexity parameter, $X$, which in the full TQE model represents the emergent physical properties of a universe, is here simplified to be a direct function of its initial energy endowment:

$$
X = E
$$

This is a crucial distinction from the complete E+I model. Here, a universe's potential for complexity is not influenced by any informational bias but is solely determined by its energy. The statistical summary of `fl_collapse_timeseries_E-Only.csv` confirms this mechanism, showing the nascent complexity parameter ($X$) settling precisely onto a lock-in value ($X_{lock}$) equivalent to the initial energy fluctuation. This establishes a deterministic link between energy and the fundamental properties of the universe, setting the stage for the subsequent evolutionary epochs.

***

## 2. Stability and the Energetic Goldilocks Zone

Following the collapse, each of the 10,000 simulated universes undergoes an evolutionary period where its governing parameters are tested for stability. The emergence of stable laws—a "lock-in" event—is the primary success criterion.

### 2.1 The Emergence of a Stability Window

The central finding of this control simulation is that **stability is not guaranteed and is strongly dependent on the initial energy $E$**. Of the 10,000 universes simulated, **5,598 (56.0%) achieved stability**, while only **2,725 (27.3%) proceeded to a full lock-in** of their physical laws. This confirms that energy alone is a viable but inefficient pathway to a stable cosmos.

The analysis of the `tqe_runs_E-Only.csv` data reveals a distinct **"Goldilocks Zone"** for stability. While the simulation's energy values ($E$) were sampled from a broad lognormal distribution (mean $\ln(E) = 2.5$), the universes that achieved stability are not uniformly distributed across this range. Instead, they cluster within an energetic band, as conceptually illustrated in the `stability_distribution_three_E-Only.png` figure. Universes with very low or extremely high energy values failed to stabilize, dissolving back into quantum chaos. This outcome empirically supports the fine-tuning function central to the TQE model, which in this simplified case becomes:

$$
P'(\psi) = P(\psi) \cdot f(E)
$$

Here, the probability of a state persisting is modulated solely by an energetic fine-tuning function, $f(E) = \exp\left(-\frac{(E-E_c)^2}{2\sigma^2}\right)$, where the simulation parameters were set to $E_c=4.0$ and $\sigma=4.0$. The results align with this principle, showing that stability is a non-monotonic function of energy, peaking within a favored range and declining outside of it.

### 2.2 Simplification of the Selection Principle

A key postulate of the full TQE model is a **two-level selection principle**, where the complexity parameter $X$ is selected for stability, while the dissonance between energy and information, $|E-I|$, determines the ease of lock-in. In the E-Only model, this dual mechanism collapses. Since $I=0$ and $X=E$, both stability and lock-in are governed by the same criterion: whether the universe's energy $E$ falls within the Goldilocks window.

This simplification is reflected in the lock-in dynamics. A universe first achieves a stable state (relative parameter change $< 0.01$ for 8 epochs) before attempting to lock-in (relative parameter change $< 0.005$ for 6 epochs). The data shows that roughly half of the stable universes (2,725 out of 5,598) successfully transition to a locked-in state, indicating that even within the stability zone, achieving permanent, unchanging laws is an additional probabilistic challenge. However, unlike the E+I model, this challenge is not mediated by an independent informational parameter but is likely a stochastic outcome related to the universe's specific trajectory within the energetic stability landscape.

***

## 3. The Archetype of an Energy-Only Universe

By analyzing the characteristics of the most successful universes (those that achieved early lock-in and maintained stability), we can construct an archetype for a cosmos formed purely by energetic principles.

The evolution of entropy in these universes, conceptually depicted in the `best_universes` plots, consistently demonstrates the emergence of a thermodynamic **arrow of time**. Following lock-in, the entropy of these systems, which was maximal post-superposition, begins a steady decline, indicating the formation of structure and order. This is a fundamental success of the model, as it reproduces a key feature of complex universes.

However, the analysis supports the hypothesis that **"In E-Only universes, rapid stabilization results in simpler, more rigid, and less dynamic worlds."** The selection mechanism is one-dimensional; it filters for a specific energy range but lacks the secondary, information-based filter that could select for more nuanced or complex configurations. The resulting stable universes are therefore archetypes of simplicity. Their laws are fixed early and remain unchanged, with their evolution driven by deterministic energetic pathways rather than a dynamic interplay between energy and information. This suggests that while E-Only universes can exist, they may represent a "minimal" class of cosmos, lacking the potential for the richer, more adaptive complexity seen in the full TQE model.

***

## 4. Testable Predictions: Cosmological Anomalies as Statistical Flukes

The TQE framework proposes that large-scale anomalies in the Cosmic Microwave Background (CMB), such as the "Axis of Evil" (AoE) or the Cold Spot, can be interpreted as relics of the law-stabilization process. The E-Only simulation provides a crucial baseline for testing this prediction.

The results show that such large-scale coherent anomalies are **exceedingly rare** in the absence of an information parameter. The `metrics_joined_E-Only` data reveals that the flag for an AoE-type alignment (`aoe_flag`) has a mean value of just **$3.0 \times 10^{-4}$**, indicating that only a handful of the 10,000 universes exhibited this feature. The `cmb_aoe_summary_E-Only.csv` file confirms this, detailing just three universes with significant alignments.

Crucially, there appears to be **no significant correlation between the emergence of these anomalies and the "quality" or stability of the universe**. They manifest as random statistical flukes rather than a feature preferentially selected for. This finding is a cornerstone of the discussion:
-   In the E-Only model, anomalies are statistical noise.
-   In the full TQE model, the information parameter $I$ is hypothesized to actively bias outcomes toward such large-scale coherent structures.

Therefore, the scarcity of anomalies in this control run acts as a **null hypothesis**. If observed cosmological data reveals statistically significant alignments beyond random chance, it would challenge the E-Only model and lend strong support to the necessity of an information parameter, making the TQE model falsifiable.


---

### Methodological Note on the Analytical Workflow

The primary conclusions presented in this document are derived from the direct statistical analysis of the simulation’s output data. This includes the aggregate statistics from the summary_full.json, the per-file checks from the Wolfram math_check.json, as well as the detailed run-level data contained in tqe_runs_E-Only.csv. This direct approach yielded robust and consistent findings.

In addition to the direct analysis, an extensive suite of predictive machine learning models was developed to probe the system's dynamics (the XAI module). While this framework is functional, the models themselves exhibited significant limitations, including overfitting and internally inconsistent explanations (as shown by conflicting SHAP and LIME results). For this reason, the conclusions from these predictive models are considered preliminary and have been excluded from the main findings of this initial publication, representing an area for future research.

The interpretation and articulation of the final analysis were performed in collaboration with several Large Language Models. An initial analysis attempt using a local DeepSeek model resulted in significant quantitative hallucinations. The final, validated analyses presented here were therefore developed and cross-checked through an iterative, Socratic dialogue with **Google's Gemini 2.5 Pro** and **OpenAI's GPT-5**, which acted as scientific reasoning and writing assistants based on the verified, factual data from the simulation.

----

## License
This project is licensed under the MIT License – see the [LICENSE](../../LICENSE) file for details.

## Contact

Got questions, ideas, or feedback?  
Drop me an email at **tqe.simulation@gmail.com** 
    
[RESULTS](../../RESULTS)

