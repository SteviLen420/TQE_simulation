SPDX-License-Identifier: MIT

Copyright (c) 2025 Stefan Len

[![CI](https://github.com/SteviLen420/TQE_simulation/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/SteviLen420/TQE_simulation/actions/workflows/ci.yml)  
[![Python](https://img.shields.io/badge/python-3.9%20|%203.10%20|%203.11-blue)](https://www.python.org/doc/)  

# TQE E+I Universe Analysis (Run ID: 20250918_081329)
**Global stability, entropy, and law lock-in metrics for Energy + Information universes**

**Author**: Stefan Len


This document summarizes the key findings from the TQE E+I simulation run `20250918_081329`. The analysis explores the conditions required for universe stability and the emergence of physical laws based on the interplay of Energy (E) and Information (I).

----------

### Figure 1: The Distribution of Universe Fates

This bar chart provides a statistical census of the final outcomes for the entire ensemble of 10,000 simulated E+I universes, categorizing them into three mutually exclusive fates.

<img width="1511" height="1232" alt="stability_distribution_three_E+I" src="https://github.com/user-attachments/assets/67e28b4d-5c40-4f10-9da4-150f9eb4d947" />

**Analysis:** 

The chart displays the distribution of the three possible end-states:

1.  **Unstable:** **5,125 universes (51.2%)** remained in a chaotic, disordered state. This is the most common outcome.
  
2.  **Stable (no lock-in):** **2,797 universes (28.0%)** successfully stabilized but their laws did not "freeze," remaining dynamic.
   
3.  **Lock-in:** **2,078 universes (20.8%)** not only became stable but also reached a final, fixed state of physical laws. This is the rarest but most favorable state for the formation of complex structures.

----------

### Figure 2: Identification of the "Goldilocks Zone" 
This plot details the relationship between a universe's initial Complexity (`X = E·I`) and its subsequent probability of reaching a favorable, structured outcome. The data is grouped into bins based on the X value, with a spline curve fitted to show the clear trend.

<img width="1209" height="842" alt="A graph showing the Goldilocks zone for universe stability" src="https://github.com/user-attachments/assets/1b97e1fc-cce3-44d6-89aa-0bb942db0d51" />

**Analysis:**

The spline fit provides compelling visual evidence for a finely-tuned "Goldilocks Zone" necessary for creating viable universes.

1. **Dynamic vs. Optimal Zone**: The simulation pipeline initially operated with a broad, dynamically-calculated "Goldilocks" window spanning from `X_low` = `1.78` to `X_high` = `39.79`. The analysis shown in this plot refines this initial estimate, identifying a much narrower optimal performance window.

2. **Refined Optimal Window**: This more precise Goldilocks Zone is marked by the vertical dashed lines, located between an `X` value of `16.40` and `27.43`. Universes within this specific range exhibit the highest probability of success.

3. **Peak Stability Probability:** The analysis pinpoints the optimal complexity for *achieving stability* at `X` ≈ `24.35`. At this value, the probability of a universe becoming stable reaches its maximum of approximately **60%**, as shown by the peak of the main curve.

4. **Conditional Lock-in Behavior:** For universes that have *already* become stable, the green curve (`P(Lock-in` | `Stable)`) reveals a different trend. The likelihood of these stable universes proceeding to a full lock-in state continues to increase with complexity, approaching rates as high as **95%** for very complex systems.

5. **Key Insight**: This creates a crucial tension: while moderate complexity is best for achieving initial stability, higher complexity appears to be more conducive to finalizing the laws of physics via lock-in once stability is present.

----------

### Figure 3: Mapping the Island of Stability in the E-I Parameter Space

This scatter plot visualizes the initial parameter space for all 10,000 simulated universes. Each point is plotted according to its initial **Energy (E)** and **Information (I)** values. The color indicates the universe's final fate: **stable (red)** or **unstable (blue)**.

<img width="1066" height="981" alt="scatter_EI_E+I" src="https://github.com/user-attachments/assets/1a820eb5-3f5f-43ed-b663-ab9f722c7651" />

**Analysis:**

The distribution of outcomes is not random, revealing a distinct **"island of stability"** in the parameter space.

1. **Concentrated Stability**: Stable universes (red dots) are overwhelmingly concentrated in a region of low-to-moderate Energy (roughly `E < 75`) and moderate-to-high Information (roughly `I > 0.2`).

2. **High-Energy Instability**: As the Energy parameter increases significantly (`E > 75`), stability becomes exceedingly rare. Universes with very high initial energy are almost universally fated to remain unstable, regardless of their Information content.

3. **The Necessity of Information**: Similarly, universes with very low Information (`I < 0.2`) have a very low probability of becoming stable, even at low energy levels.

**Key Insight**: This plot strongly suggests that stability is an emergent property arising from a **co-dependent relationship between E and I**. A sufficient level of Information appears necessary to structure a universe and guide it toward stability. However, this structuring effect seems to fail when the system's energy is overwhelmingly high, leading to inevitable chaos. This visualization provides a complementary view to the "Goldilocks Zone," showing how the individual components of `X` contribute to a universe's viability.

----------

### Figure 4: The Initial Quantum Fluctuation Phase

This plot shows the behavior of a key observable (`A`) during the initial quantum fluctuation stage. This phase models the quantum "jitter" from which the initial conditions of each universe emerge. The chart displays the **expectation value (mean)** of `A` and its **variance** over time.

<img width="1209" height="842" alt="fl_fluctuation_E+I" src="https://github.com/user-attachments/assets/6a0f9ce2-3624-40ab-b3a8-37d5c8777d7e" />

**Analysis:**

The graph demonstrates that the simulation's initial conditions are generated in a stable and controlled manner, consistent with theoretical models of quantum fluctuations.

1. **Unbiased Mean (`⟨A⟩`)**: The expectation value of `A` (blue dashed line) rapidly oscillates around zero before quickly converging to `⟨A⟩ = 0`. This is a crucial result, indicating that the fluctuation process is unbiased and does not introduce any preferred value or direction into the nascent universe's structure.

2. **Stable Variance (`Var(A)`)**: The variance of `A`(orange solid line) immediately stabilizes at a constant value of `Var(A) = 1`. This normalization ensures that the magnitude of the initial fluctuations is consistent and well-defined for every simulation run.

**Key Insight:** The rapid convergence of these two metrics confirms that the "seed" for each universe's evolution is generated from a well-behaved, stable quantum process. This robust foundation is essential for the integrity of the subsequent simulation steps where these initial fluctuations evolve into larger structures.

----------

### Figure 5: Evolution of Entropy and Purity in Quantum Superposition

This plot illustrates the evolution of two fundamental quantum metrics, **Entropy** and **Purity**, during the initial superposition phase (`t < 0`). These inversely correlated measures describe the quantum state of the nascent universe before it collapses into a classical reality.

<img width="1209" height="842" alt="fl_superposition_E+I" src="https://github.com/user-attachments/assets/798efc54-5e8c-4ef3-aecc-898d61f3ddbb" />

**Analysis:**

The chart captures the critical transition from a simple initial state to one of rich quantum potential.

1. **Initial State**: At `t = 0`, the system begins in a well-defined state of **high Purity** (approaching 1) and **low Entropy** (approaching 0). This represents a simple, unmixed quantum origin.

2. **Rapid Decoherence**: The system undergoes an extremely rapid evolution. The Purity (orange line) plummets, while the Entropy (blue line) skyrockets, with both metrics reaching a stable equilibrium by approximately `t = 2.0`.

3. **Maximally Mixed State**: The system settles into a state of **maximal Entropy** (fluctuating around 1.0) and minimal Purity (fluctuating around 0.2-0.3). This is the signature of a maximally mixed state, where the universe exists in a rich superposition of all its possible basis states.

**Key Insight**: This process is fundamental to the TQE model. It ensures that each universe begins without a predisposition toward any specific outcome. By evolving into a state of maximum superposition, the simulation creates a level playing field of possibilities. From this state, a single classical reality is later selected or "collapses," driven by the specific E and I parameters of that universe.

------------

### Figure 6: The Collapse Event - From Quantum Potential to Physical Reality

This plot visualizes the pivotal "collapse" event at `t=0`. It shows the behavior of the **Complexity parameter (X)** for a single representative universe as it transitions from a fluctuating quantum state (`t < 0`) to a stable, classical state (`t > 0`).

<img width="1219" height="842" alt="fl_collapse_E+I" src="https://github.com/user-attachments/assets/e4312dbe-9d86-4f67-a945-a15a6be7d20a" />

Analysis:
This graph captures the instant a universe's fundamental properties are determined.

1. **Pre-Collapse Superposition (`t < 0`)**: Before the collapse, the `X` value (gray line) is highly volatile, exhibiting large and chaotic fluctuations. This represents the universe existing in a superposition of many different potential complexity values, with no single defined reality.

2. **The Collapse Event (`t = 0`)**: The vertical red line marks the moment of collapse. At this instant, one specific value for X is "selected" from the quantum foam of possibilities, defining the physical nature of this particular universe.

3. **Post-Collapse Lock-in (`t > 0`)**: Immediately following the collapse, the chaotic fluctuations cease entirely. The `X` parameter rapidly stabilizes, settling into minor oscillations around a fixed value of `X = 3.77` (indicated by the dashed red line). This becomes the locked-in value of Complexity that will govern the universe's subsequent classical evolution.

**Key Insight:** The collapse event is the central mechanism in the TQE simulation that transforms quantum indeterminacy into a classical, predictable reality. The specific value that `X` locks into is critical; its position within or outside the "Goldilocks Zone" is a primary determinant of the universe's long-term fate and its potential to become stable.

-----------
