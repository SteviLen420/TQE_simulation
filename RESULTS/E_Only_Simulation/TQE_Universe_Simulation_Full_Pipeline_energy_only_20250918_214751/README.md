SPDX-License-Identifier: MIT

Copyright (c) 2025 Stefan Len

[![CI](https://github.com/SteviLen420/TQE_simulation/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/SteviLen420/TQE_simulation/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.9%20|%203.10%20|%203.11-blue)](https://www.python.org/doc/)

# TQE Simulation Results for E-Only Universe (Run ID: 20250918_214751)
### Author: Stefan Len


This document presents a comprehensive analysis of the numerical simulation for the Theory of the Question of Existence (TQE), specifically focusing on the **E-Only control run**. In this configuration, the information orientation parameter ($I$) is set to zero, allowing for an isolated examination of the role of vacuum energy ($E$) in the cosmogenesis and stabilization of physical law. The simulation encompassed **10,000 universes**, each evolving over **3,000 epochs**.

---

### 1. The Initial Phases of Cosmogenesis: From Fluctuation to Collapse

The TQE model posits that a universe originates from a quantum fluctuation in a pre-law state. Our simulation operationalizes this by modeling three distinct initial phases: quantum fluctuation, superposition, and collapse.

- **Fluctuation and Superposition:** The simulation begins by generating a vast landscape of potential physical realities. Data from the superposition phase (`fl_superposition_timeseries`) confirms that the system rapidly evolves into a state of high entropy, reaching a mean Shannon entropy ($H$) of approximately **0.96**. This value, close to the theoretical maximum of 1.0, represents a state of maximal potential where no specific outcome is favored—a statistical "blank slate." This phase successfully models the creation of an unbaised possibility space before the emergence of any defined physical law.

- **Collapse and Parameter Fixation:** The subsequent collapse phase forces this superposition into a definite state. In this E-Only control model, the complexity parameter ($X$), which underpins the emergent physical laws, is determined solely by the initial energy fluctuation. This establishes the foundational equation for this simulation variant:

$$
X = E
$$

This direct equivalence implies that all subsequent dynamics, including the potential for stabilization, are exclusively a function of the universe's initial energy. This contrasts sharply with the full TQE hypothesis, where complexity would arise from a coupling of both energy and information, i.e., $X = f(E, I)$. The absence of the informational parameter ($I=0$) provides a crucial baseline to isolate the effects of energy alone.

---

### 2. Mechanisms of Stability: The Emergence of a "Goldilocks Zone"

Following the initial collapse, each universe evolves over 3,000 epochs, with its stability contingent on whether its parameters can "lock-in." The simulation's primary output reveals that stable laws are not a generic outcome but are confined to a specific energetic range.

- **Overall Stability Rates:** Of the 10,000 universes simulated, **5,451 (54.5%)** achieved a state of preliminary stability, while only **2,516 (25.2%)** met the stringent "lock-in" criterion, defined as a relative parameter change $\Delta P / P < 0.005$ over 6 consecutive epochs. This demonstrates that stabilization is a non-trivial process even under simplified, energy-only conditions.

- **The Goldilocks Window:** Analysis of the raw data reveals a clear "Goldilocks" effect. Universes with very low initial energy ($E \lesssim 4.0$) consistently failed to stabilize. As energy increases, the probability of achieving both stability and lock-in rises significantly. The simulation identified a dynamic stability window for the complexity parameter $X$ between **4.74 and 97.12**. Since $X=E$ in this model, this directly translates to an energetic Goldilocks window. This finding contradicts the possibility of a simple, monotonic relationship between energy and stability, instead pointing to a threshold-based mechanism where a minimum energy level is required to initiate and sustain stable laws.

- **Simplification of the Selection Principle:** The TQE framework proposes a two-level selection process: one for stability ($X$) and another for constancy ($|E-I|$). In the E-Only model, with $I=0$, this distinction collapses. The probability update rule simplifies from $P'(\psi) = P(\psi) \cdot f(E,I)$ to:

$$
P'(\psi) = P(\psi) \cdot f(E)
$$

Here, the fine-tuning function $f(E)$ depends only on the energy's proximity to a critical value ($E_c$):

$$
f(E) = \exp\left(-\frac{(E-E_c)^2}{2\sigma^2}\right)
$$

Consequently, both the initial selection of laws and their subsequent stabilization are governed by a single factor: the magnitude of the initial energy fluctuation.

---

### 3. The Archetype of a Successful Universe in an E-Only Cosmos

The TQE framework hypothesizes that while energy alone may be sufficient to generate stable laws, the resulting universes would lack the finely-tuned complexity required for life, which is proposed to emerge from the non-trivial interplay of both Energy and Information.

The results from this E-Only simulation do not and cannot prove this hypothesis, as no E+I data is available for comparison. However, these findings provide a critical **baseline** and **theoretical foundation** for its future investigation. The simulation demonstrates that a significant fraction of universes (25.2%) can achieve law-stabilization through purely energetic mechanisms. This cohort of "successful" E-Only universes serves as the ultimate control group.

Future research involving E+I simulations can now be framed against these results. Key questions will include:
1.  Does the introduction of $I$ significantly alter the rate of lock-in?
2.  Do the emergent laws in E+I universes exhibit different statistical properties compared to their E-Only counterparts?
3.  Does the $|E-I|$ parameter provide a more robust selection mechanism for creating universes with the specific kind of complexity observed in our own?

Thus, the current analysis **sharpens the core hypothesis of TQE** by demonstrating what energy alone can accomplish, thereby setting the stage to investigate what it cannot.

---

### 4. Testable Predictions: The Rarity and Nature of Cosmological Anomalies

A key feature of the TQE model is its potential to generate falsifiable predictions related to large-scale cosmological anomalies, such as the Cosmic Microwave Background (CMB) Cold Spot and the Axis of Evil (AoE).

The E-Only simulation successfully generated universes exhibiting these anomalies, but their occurrence was exceedingly rare. Out of 10,000 universes, only **9** instances of either a Cold Spot or an AoE were recorded, corresponding to a rate of less than **0.1%**.

Crucially, an analysis of these 9 anomalous universes reveals a profound connection to stability. All of them were found within the subset of 2,516 universes that had successfully achieved **lock-in**. Their mean energy ($E=45.6$) and lock-in epoch ($\text{epoch}=306$) place them firmly within the stable Goldilocks zone. This suggests that in the TQE framework, large-scale anomalies are not random artifacts but are instead **rare, emergent features of otherwise stable, law-abiding universes**. They can be interpreted as "fossils" of the violent, chaotic stabilization process itself. This provides a clear, testable prediction: a statistical correlation should exist between the presence of large-scale CMB anomalies and other markers of a finely-tuned, stable cosmos.




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
