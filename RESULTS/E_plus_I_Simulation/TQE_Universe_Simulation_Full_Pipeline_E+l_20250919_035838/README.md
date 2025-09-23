SPDX-License-Identifier: MIT

Copyright (c) 2025 Stefan Len

[![CI](https://github.com/SteviLen420/TQE_simulation/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/SteviLen420/TQE_simulation/actions/workflows/ci.yml)  
[![Python](https://img.shields.io/badge/python-3.9%20|%203.10%20|%203.11-blue)](https://www.python.org/doc/)  

# TQE E+I Universe Analysis (Run ID: 20250919_035838)
**Global stability, entropy, and law lock-in metrics for Energy + Information universes**

**Author**: Stefan Len


This document summarizes the key findings from the TQE E+I simulation run `20250918_035838`. The analysis explores the conditions required for universe stability and the emergence of physical laws based on the interplay of Energy (E) and Information (I).

----------

### Figure 1: Distribution of Simulated Universe Fates in the E+I Cohort

This bar chart illustrates the final distribution of outcomes for the **10,000 simulated universes** in the E+I cohort, where both Energy (E) and Information (I) are active parameters. The universes are classified into three distinct categories based on their long-term behavior: achieving "Lock-in," achieving stability without lock-in, or remaining unstable.

<img width="1511" height="1232" alt="stability_distribution_three_E+I" src="https://github.com/user-attachments/assets/c5be9887-08a3-4910-9972-abfab01dc85f" />

**Analysis:**

The results provide a foundational overview of the ultimate fates of universes governed by the interplay of Energy and Information.

1. **Prevalence of Instability**: The most common outcome was instability, with **4,907 universes (49.1%)** failing to reach a stable state. This suggests that the conditions for developing a consistent, ordered cosmos are not met in approximately half of the cases, even with the guiding influence of Information.

2. **Overall Stability Rate**: A slight majority of universes, **5,093 in total (50.9%)**, achieved some form of stability. This cohort is further divided into two sub-categories, indicating different degrees of cosmic resolution.

3. **Hierarchy of Stability**: Within the stable group, **2,884 universes (28.8%)** reached a "Stable (no lock-in)" state, where their fundamental parameters ceased to fluctuate significantly. A smaller but substantial subset of **2,209 universes (22.1%)** achieved the stronger condition of "Lock-in," where the physical laws themselves became permanently fixed.

------------

### Figure 2: The Goldilocks Zone for Universe Stability

This plot reveals the relationship between the initial Complexity parameter `(X = E·I)` of a simulated universe and its resulting probability of achieving a stable outcome. The analysis is based on the `10,000` universes from the E+I cohort. The blue points represent the mean stability rate for universes grouped into discrete bins by their X-value, while the red spline fit illustrates the overall trend.

<img width="1209" height="842" alt="stability_curve_E+I" src="https://github.com/user-attachments/assets/910b0e05-0fef-41ac-88c3-4ecd1886f0e2" />

**Analysis:**

The spline curve provides compelling quantitative evidence for a precisely defined "Goldilocks Zone"—a narrow range of initial complexity that is highly conducive to forming a stable universe.

1. **Optimal Complexity Range**: The analysis identifies a well-defined optimal zone for stability, indicated by the vertical dashed lines. Universes with an initial complexity value between `X = 23.09` and `X = 27.21` have the highest likelihood of evolving into a stable state.

2. **Peak Stability**: The curve reaches its global maximum at an initial complexity of `X = 25.56`. At this specific value, the probability of a universe achieving stability is maximal, approaching approximately 95%.

3. **High Sensitivity**: The probability of stability drops sharply outside of the optimal zone. Universes with too little complexity (`X < 20`) or too much (`X > 30`) are significantly less likely to become stable. This highlights the fine-tuned nature of this parameter. A smaller, secondary peak is observed near `X ≈ 32`, but it represents a much lower probability of success.

**Key Insight**: This figure powerfully demonstrates that a universe's fate is critically sensitive to the initial balance between its Energy and Information content, as quantified by the Complexity parameter `X`. The existence of a sharp, narrow peak underscores the core tenet of fine-tuning; it is not merely the presence of E and I, but their specific multiplicative relationship that is paramount. Both an excess and a deficit of this combined "complexity factor" are overwhelmingly detrimental to the formation of a stable cosmos within the TQE framework.

--------------

### Figure 3: Stability Outcomes in the Energy-Information Parameter Space

This scatter plot visualizes the initial parameter space for all 10,000 universes simulated in the E+I cohort. Each point represents a single universe, positioned according to its initial **Energy (E)** on the x-axis and **Information (I)** on the y-axis. The color of each point indicates its final evolutionary outcome, as shown by the color bar: **red for stable universes (1) and blue for unstable universes (0)**.

<img width="1066" height="981" alt="scatter_EI_E+I" src="https://github.com/user-attachments/assets/95b974cd-5aaf-4a60-bce2-363f587d75f5" />

Analysis:

The plot reveals the complex, non-linear relationship between the initial `E` and `I` parameters and the likelihood of a universe achieving stability.

1. **Parameter Distributions**: The initial conditions show a distinct distribution pattern. The Energy (E) parameter is heavily skewed toward lower values (most universes have `E < 50`), with a long tail of high-energy outliers, which is characteristic of a log-normal distribution. The Information (I) parameter is more evenly distributed across its range.

2. **Low-Energy Regime**: In the region of low energy (approximately `E < 50`), where the vast majority of universes are instantiated, there is a dense intermixing of both stable (red) and unstable (blue) outcomes. This indicates that for low-E universes, the Energy value alone is a poor predictor of stability; the outcome is critically dependent on the corresponding Information value.

3. **High-Energy Trend**: A clear trend emerges at higher energy levels (roughly `E > 75`). Unstable (blue) universes become exceedingly rare in this regime. Although fewer universes are generated with such high initial energy, those that are have a very high probability of evolving into a stable state.

4. **Implicit Goldilocks Correlation**: While this plot does not explicitly graph the Complexity `X = E·I`, the concentration of stable red points forms a discernible pattern. This pattern is consistent with the hyperbolic curve (`I = constant / E`) that would define the optimal "Goldilocks Zone" identified in Figure 2. The most successful universes are not found at the extremes of either E or I, but within a specific combinatoric region that balances the two.

**Key Insight**: This visualization powerfully illustrates that cosmic stability in the TQE model is not a simple monotonic function of either Energy or Information. While very high energy appears to be a sufficient condition for stability, it is a rare initial state. For the majority of universes born in the more common low-energy regime, a delicate balance with the Information parameter is required to achieve an ordered outcome. This plot decomposes the abstract "Goldilocks Zone" of Complexity into its constituent parts, reinforcing the central thesis that it is the **interplay and fine-tuned balance between E and I**, not just their individual magnitudes, that fundamentally determines a universe's potential for a stable existence.

---------------

### Figure 4: Statistical Evolution During the Quantum Fluctuation Phase

This plot displays the time evolution of the expectation value (`⟨A`⟩, mean) and variance (`Var(A)`) of a fundamental observable 'A' during the initial quantum fluctuation stage of the simulation. These curves represent the averaged behavior across the entire 10,000-universe E+I cohort, providing insight into the primordial state from which each universe emerges.

<img width="1230" height="842" alt="fl_fluctuation_E+I" src="https://github.com/user-attachments/assets/e93edd4f-dd24-4831-905f-898afdb16658" />

**Analysis:**

The graph demonstrates a swift and decisive process of initial state-setting common to all simulated universes.

1. **Rapid Convergence**: The most prominent feature is the rapid stabilization of both statistical moments. Within the first unit of simulation time (`t < 1.0`), both the expectation value and the variance converge to stable equilibrium values, after which they remain constant.

2. **Zero-Mean Expectation Value**: The expectation value `⟨A⟩` (dashed blue line), following a brief initial dip, quickly converges to and holds a value of **zero**. This indicates that the quantum fluctuations are, on average, unbiased, producing no net positive or negative value for the observable A. This is consistent with an origin from pure, directionless quantum noise.

3. **Normalized Variance**: The variance `Var(A)` (solid orange line) begins at a high value (`≈ 0.95`) and rapidly converges to a stable value of **one**. A system that resolves to a state with a mean of `0` and a variance of `1` is characteristic of a standardized random variable, suggesting the fluctuation phase establishes a normalized and statistically well-behaved foundation.

**Key Insight**: This figure illustrates the critical function of the quantum fluctuation phase: to serve as a robust initialization mechanism. It ensures that every universe, regardless of its specific E and I parameters, begins its evolutionary journey from a consistent and statistically standardized state, analogous to a normalized vacuum. The rapid convergence implies that this state-setting is a foundational and highly efficient process, creating a stable "blank slate" of unbiased randomness from which the more complex, parameter-dependent evolution toward stability or instability can subsequently unfold.

-----------------

### Figure 5: Evolution of Entropy and Purity during the Quantum Superposition Phase

This graph plots the average behavior of two fundamental quantum-informational metrics—**Entropy** (blue dashed line) and **Purity** (orange dashed line)—over time during the quantum superposition phase. This stage models the universe as a complex superposition of states, preceding the collapse into a single, classical reality.

<img width="1209" height="842" alt="fl_superposition_E+I" src="https://github.com/user-attachments/assets/9a7fa42c-7570-49b2-9f08-924b9088614e" />

**Analysis:**

The plot illustrates a rapid and dramatic transformation from a state of perfect order to one of maximal complexity and uncertainty.

1. **Inverse Correlation**: A stark inverse relationship exists between Entropy and Purity. At `t=0`, the system begins in a state of maximum Purity (1.0) and minimum Entropy (near 0). As the simulation progresses, Purity rapidly declines while Entropy reciprocally increases.

2. **Entropy Saturation**: The Entropy of the system skyrockets from nearly zero to a maximum value of approximately **1.0** within the first 1.5 units of time. After this point, it fluctuates around this maximal value, indicating the system has reached a state of maximum statistical mixedness or uncertainty.

3. **Purity Collapse**: Correspondingly, the Purity of the state begins at 1.0 (a perfectly pure state) and collapses to a fluctuating low value of around 0.25. This demonstrates that the initial, well-defined state has evolved into a highly mixed state, representing a complex superposition of numerous potential realities.

**Key Insight**: This figure models the creation of the quantum "state-space" for the nascent universes. The simulation initializes each cosmos in a simple, pure state of low information content (low entropy). The superposition process then rapidly evolves this into a maximally mixed state, representing a rich sea of possibilities where all potential outcomes coexist. This high-entropy, low-purity equilibrium is a critical step, establishing the quantum substrate from which a single, classical universe is selected during the subsequent "collapse" phase of its evolution. The speed of this transition suggests it is a foundational and universal feature of cosmic genesis in the TQE model.

----------------

### Figure 6: The Collapse Event and Initial Parameter Fixation of X

This time-series plot illustrates the pivotal "collapse" event, centered at `t=0`. It shows the universe's fundamental Complexity parameter (`X = E·I`) transitioning from a volatile, indeterminate quantum phase to a stable, classical phase with a single, definite value. The gray line tracks the value of `X`, averaged for a cohort of universes undergoing this transition.

<img width="1243" height="842" alt="fl_collapse_E+I" src="https://github.com/user-attachments/assets/0776c0ec-4ecd-4e3c-b8bf-6b3310442ece" />

**Analysis:**

The graph provides a clear depiction of the universe's transition from a multi-potential quantum state to a singular classical reality.

1. **Pre-Collapse Superposition (`t < 0`)**: Before the event at `t=0`, the Complexity `X` lacks a definite value, exhibiting large and erratic fluctuations. This represents the quantum superposition state where a wide range of potential `X` values coexist.

2. **The Collapse Instant (`t = 0`)**: Marked by the vertical red line, this is the moment of decoherence. The superposition of states resolves, and a single, specific value for the Complexity parameter is selected.

3. **Post-Collapse Parameter Fixation (`t > 0`)**: Immediately following the collapse, the system's behavior changes fundamentally. The large-scale quantum fluctuations vanish and are replaced by low-amplitude, classical-like oscillations around a now-stable mean.

4. **Resulting `X` Parameter**: The horizontal red dashed line indicates the specific value to which X has collapsed. For this representative case, the universe proceeds its evolution governed by the fixed fundamental parameter of `X = 3.70`.

**Key Insight**: This figure models the crucial step of parameter fixation, the TQE model's analog for a quantum measurement or wave function collapse. This event is the necessary precondition for, but distinct from, the later evolutionary state of "Law Lock-in." At `t=0`, the universe's fundamental nature, defined by its `X` value, becomes fixed. This newly established constant then governs the universe's entire subsequent evolution during the expansion phase. Whether that evolution will ultimately lead to the final, immutable state of "Law Lock-in" is determined by the specific value of `X` that was selected in this primordial collapse event.

------------------

### Figure 7: Expansion Dynamics and the Average Epoch of Law Lock-in

This plot illustrates the post-collapse dynamics for the E+I universe cohort, focusing on the expansion phase from `epoch 0` to `800`. The graph tracks the evolution of the universe's expansion (`Amplitude A`) and a secondary `Orientation I` parameter. Crucially, it marks the average epoch at which a subset of these universes achieves the final "Law Lock-in" state.

<img width="1390" height="842" alt="fl_expansion_E+I" src="https://github.com/user-attachments/assets/b1c463f2-56dc-42aa-84d1-1bc7b3b764dc" />

**Analysis:**

The chart showcases the primary long-term evolutionary behavior of the simulated universes after their fundamental `X` parameter has been fixed.

Sustained Cosmic Expansion: `The Amplitude A` (blue line) demonstrates a clear and persistent growth trend, representing the ongoing expansion of the universe's scale. The expansion appears roughly linear with small stochastic fluctuations, indicating a steady increase in size over time.

Parameter Decoupling: The `Orientation I` parameter (orange line) remains stable and close to zero throughout the entire epoch. This suggests that this particular parameter is decoupled from the expansion dynamics, having either been fixed at `t=0` or being a conserved quantity within the model.

Late-Stage Law Lock-in: The vertical red dashed line identifies the average epoch for "Law Lock-in" at approximately `t = 747`. Based on the provided context, this is a mean value calculated only from the subset of universes (22.1% of the total) that successfully reached this ultimate state of stability. This confirms that Law Lock-in is not an initial condition but an emergent state achieved late in a universe's evolution.

Independence of Expansion and Finality: A critical observation is that the universe's expansion (`Amplitude A`) continues unabated through and beyond the point of Law Lock-in. The freezing of the universe's fundamental physical rules does not halt or alter the metric expansion of spacetime.

**Key Insight**: This figure provides a crucial distinction between a universe's dynamic evolution and the finalization of its laws. The "Law Lock-in" event is shown to be a late-stage achievement for a fraction of universes, not a universal starting point. The most significant insight is the **decoupling of physical law finality from spatial expansion**. In the TQE model, a universe can solidify its fundamental constants and rules while its spatial fabric continues to expand. This suggests that a universe's core identity (`laws`) and its dynamic behavior (`expansion`) are distinct and can operate on different evolutionary timescales.

-------------------
