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

**Analysis:** The chart displays the distribution of the three possible end-states:
1.  **Unstable:** **5,125 universes (51.2%)** remained in a chaotic, disordered state. This is the most common outcome.
2.  **Stable (no lock-in):** **2,797 universes (28.0%)** successfully stabilized but their laws did not "freeze," remaining dynamic.
3.  **Lock-in:** **2,078 universes (20.8%)** not only became stable but also reached a final, fixed state of physical laws. This is the rarest but most favorable state for the formation of complex structures.

### Figure 2: Identification of the "Goldilocks Zone" 

This plot details the relationship between a universe's initial Complexity (X = E·I) and its subsequent probability of reaching a favorable, structured outcome. The data is grouped into bins based on the `X value`, with a spline curve fitted to show the clear trend.

<img width="1209" height="842" alt="stability_curve_E+I" src="https://github.com/user-attachments/assets/1b97e1fc-cce3-44d6-89aa-0bb942db0d51" />

Rendben, ez egy lényeges pont, ami mélyebb kontextust ad az ábrának. A szimuláció által használt tágabb, dinamikus zóna és a grafikonon látható, finomított optimális zóna közötti különbség fontos részlet.

Beépítettem ezt az információt az elemzésbe, hogy még teljesebb legyen a kép.

Figure 2: Identification of the "Goldilocks Zone" (Kibővített elemzés)
This plot details the relationship between a universe's initial Complexity (X = E·I) and its subsequent probability of reaching a favorable, structured outcome. The data is grouped into bins based on the X value (blue dots), with a spline curve (red line) fitted to show the clear trend.

<img width="1209" height="842" alt="A graph showing the Goldilocks zone for universe stability" src="https://github.com/user-attachments/assets/1b97e1fc-cce3-44d6-89aa-0bb942db0d51" />

### Analysis:
The spline fit provides compelling visual evidence for a finely-tuned "Goldilocks Zone" necessary for creating viable universes.

1. **Dynamic vs. Optimal Zone**: The simulation pipeline initially operated with a broad, dynamically-calculated "Goldilocks" window spanning from X_low = 1.78 to X_high = 39.79. The analysis shown in this plot refines this initial estimate, identifying a much narrower optimal performance window.

2. **Refined Optimal Window**: This more precise Goldilocks Zone is marked by the vertical dashed lines, located between an X value of 16.40 and 27.43. Universes within this specific range exhibit the highest probability of success.

3. **Peak Stability Probability:** The analysis pinpoints the optimal complexity for *achieving stability* at **X ≈ 24.35**. At this value, the probability of a universe becoming stable reaches its maximum of approximately **60%**, as shown by the peak of the main curve.

4. **Conditional Lock-in Behavior:** For universes that have *already* become stable, the green curve (`P(Lock-in | Stable)`) reveals a different trend. The likelihood of these stable universes proceeding to a full lock-in state continues to increase with complexity, approaching rates as high as **95%** for very complex systems.

5. **Key Insight**: This creates a crucial tension: while moderate complexity is best for achieving initial stability, higher complexity appears to be more conducive to finalizing the laws of physics via lock-in once stability is present.

