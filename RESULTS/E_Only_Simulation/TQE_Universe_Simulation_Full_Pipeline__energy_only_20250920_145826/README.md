SPDX-License-Identifier: MIT

Copyright (c) 2025 Stefan Len

[![CI](https://github.com/SteviLen420/TQE_simulation/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/SteviLen420/TQE_simulation/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.9%20|%203.10%20|%203.11-blue)](https://www.python.org/doc/)

# TQE Simulation Results for E-Only Universe (Run ID: 20250920_145826)
### Author: Stefan Len


This document presents a comprehensive analysis of the control simulation for the Theory of the Question of Existence (TQE), termed the "Energy-Only" or "E-Only" model. In this configuration, the information orientation parameter ($I$) is set to zero ($I=0$) for all 10,000 simulated universes. The objective is to investigate the emergence of stable physical laws from quantum fluctuations driven solely by an energetic component, thereby establishing a baseline against which the full Energy-Information coupling model can be compared.

---

## 1. Cosmogenesis: From Quantum Fluctuation to a Defined State

The TQE framework posits that cosmogenesis initiates from a quantum fluctuation in a pre-law state, where the universe exists as a superpositional probability distribution $P(\psi)$. The E-Only simulation models this process through distinct phases, beginning with the establishment of a normalized initial state.

### 1.1. Fluctuation, Superposition, and Collapse

The initial **quantum fluctuation phase** introduces vacuum energy ($E$) into the system, sampled from a log-normal distribution ($E_{log\_mu}=2.5$, $E_{log\_\sigma}=0.8$) to account for a heavy tail, allowing for rare, high-energy events. This energy acts as the sole driver of complexity, as per the simplified complexity definition for this control run:
$$ X = E $$

Following the initial energy injection, the system enters a **superposition phase**. As indicated by the simulation artifacts (`fl_superposition_timeseries`), this phase is characterized by high entropy (mean entropy ≈ 0.967) and correspondingly low purity (mean ≈ 0.276). This outcome demonstrates that the model successfully generates a state of maximal uncertainty—a "possibility space" where numerous potential outcomes coexist without a predetermined structure, representing a statistically normalized "blank slate."

The subsequent **collapse phase** resolves this superposition into a single, definite state. The simulation data (`fl_collapse_timeseries`) shows the complexity parameter $X$ rapidly converging to a stable lock-in value ($X_{lock}$). Crucially, in the absence of an information parameter ($I=0$), the mechanism for state selection is purely energetic. The final state of the universe, represented by its core physical parameter $X$, is determined entirely by the initial energy fluctuation. This contrasts sharply with the full TQE model, where the interplay between $E$ and $I$ introduces a more nuanced selection process.

---

## 2. The Emergence of Stability: An Energetic Goldilocks Zone

The central test of the E-Only model is whether stable, law-abiding universes can emerge without an informational bias. The simulation results confirm that stability is indeed possible, but it is a rare outcome heavily constrained by energy.

### 2.1. Stability Statistics and Energetic Dependence

Out of 10,000 simulated universes, **4,421 (44.21%) achieved stability**, meaning their fundamental parameters ceased to fluctuate wildly. However, a stricter criterion for a viable, complexity-permitting universe is "lock-in," where the laws not only stabilize but become permanently fixed. Only **1,587 universes (15.87%) achieved this state**. This significant drop-off highlights that mere stability is insufficient; a robust mechanism is required to render physical laws immutable.

The probability of a universe achieving stability is governed by a fine-tuning function that, in this control run, simplifies to a function of energy alone:
$$ P'(\psi) = P(\psi) \cdot f(E) $$

The simulation implements this through a stochastic process where outcomes are biased toward stability only within a specific energetic range. Analysis of the stable and locked-in populations confirms this. While a simple machine learning model (fine-tune detector) achieves an AUC score of 0.617 for predicting stability based on energy features, indicating a better-than-random correlation, the relationship is not linear. The stability distribution plot (`stability_distribution.png`) and the underlying data reveal that stability does not increase monotonically with energy. Instead, universes preferentially stabilize within a specific range, an effect termed the **"Goldilocks Zone."** The simulation dynamically identified this window for the complexity parameter $X$ (and thus for $E$) to be between approximately **4.05 and 101.85**.

### 2.2. A Single-Level Selection Mechanism

In the full TQE model, a two-level selection principle is proposed: the complexity parameter $X$ is fine-tuned for stability, while the energy-information gap, $|E-I|$, is fine-tuned for permanent lock-in. With $I=0$, this dual mechanism collapses into a single-level, energy-dependent process. A universe's potential for both stability and lock-in is determined entirely by whether its initial energy fluctuation $E$ falls within the Goldilocks window. The lock-in criterion is met when the relative change in parameters falls below a threshold (${\Delta P}/{P} < 0.005$) for a sufficient duration (6 epochs), a process that is only probabilistically favored within this energetic band. The absence of the informational component ($I$) removes the secondary selection pressure that, in the main theory, is hypothesized to drive the system toward more complex and robust configurations.

---

## 3. Archetype of an Energy-Only Universe

By examining the characteristics of the most stable, "best-performing" universes, we can construct an archetype for a cosmos formed without informational guidance. The simulation artifacts tracking their evolution (`best_universes` directory) suggest a distinct developmental trajectory.

The expansion phase data (`fl_expansion_timeseries`) shows a clear, monotonic increase in the scale factor ($A$), confirming that a recognizable **arrow of time** emerges as a fundamental property of expanding, post-collapse universes, even without an informational directive. This suggests that temporal directionality is a natural consequence of the energetic expansion itself.

However, the nature of this stability appears to be qualitatively different from that hypothesized for E+I universes. Without the informational bias ($ \alpha \cdot I $) in the fine-tuning function, there is no pressure favoring states of higher complexity or more nuanced internal structure.
$$ f(E,I) = \exp\left(-\frac{(E-E_c)^2}{2\sigma^2}\right) \cdot (1 + \alpha \cdot I) \rightarrow f(E) = \exp\left(-\frac{(E-E_c)^2}{2\sigma^2}\right) $$
This supports the core hypothesis that **in E-Only universes, rapid stabilization leads to simpler, more rigid, and less dynamically structured worlds**. The laws lock into place quickly if the energy is "right," but the resulting physics lacks the potential for rich, hierarchical complexity that the information parameter is theorized to enable. These universes are stable but comparatively barren.

---

## 4. Cosmological Anomalies as Falsifiable Predictions

A key feature of the TQE framework is its ability to generate falsifiable predictions related to large-scale cosmological anomalies. The E-Only simulation was configured to model phenomena analogous to the Cosmic Microwave Background (CMB) "Axis of Evil" (AoE) and Cold Spot (CS).

The results demonstrate that such anomalies are exceptionally rare in a purely energy-driven cosmogenesis. The summary statistics (`cmb_aoe_summary`) show that only **3 out of the 10,000 universes (0.03%)** exhibited a statistically significant Axis of Evil alignment. This rarity suggests that such large-scale coherent structures are not a generic outcome of random quantum fluctuations.

Furthermore, the anomalous universes are not random samples from the general population. Their mean energy ($E \approx 42.9$) is significantly higher than the population mean ($E \approx 17.0$), and they tend to lock-in their physical laws much later (mean `lock_epoch` ≈ 306 vs. ~97 for the general locked-in population). This indicates that in the TQE model, **large-scale anomalies are correlated with high-energy, late-stabilization events**. They are not statistical flukes but potential signatures of a specific, high-tension cosmogenesis pathway. This provides a clear, testable prediction that distinguishes TQE from models where such anomalies are purely random occurrences.

---

### Methodological Note on the Analytical Workflow

The primary conclusions presented in this document are derived from the direct statistical analysis of the simulation’s output data. This includes the aggregate statistics from the summary_full.json, the per-file checks from the Wolfram math_check.json, as well as the detailed run-level data contained in tqe_runs_E-Only.csv. This direct approach yielded robust and consistent findings.

In addition to the direct analysis, an extensive suite of predictive machine learning models was developed to probe the system's dynamics (the XAI module). While this framework is functional, the models themselves exhibited significant limitations, including overfitting and internally inconsistent explanations (as shown by conflicting SHAP and LIME results). For this reason, the conclusions from these predictive models are considered preliminary and have been excluded from the main findings of this initial publication, representing an area for future research.

The interpretation and articulation of the final analysis were performed in collaboration with several Large Language Models. An initial analysis attempt using a local DeepSeek model resulted in significant quantitative hallucinations. The final, validated analyses presented here were therefore developed and cross-checked through an iterative, Socratic dialogue with **Google's Gemini 2.5 Pro** and **OpenAI's GPT-5**, which acted as scientific reasoning and writing assistants based on the verified, factual data from the simulation.


## License
This project is licensed under the MIT License – see the [LICENSE](../../LICENSE) file for details.

## Contact

Got questions, ideas, or feedback?  
Drop me an email at **tqe.simulation@gmail.com** 
    
[RESULTS](../../RESULTS)
