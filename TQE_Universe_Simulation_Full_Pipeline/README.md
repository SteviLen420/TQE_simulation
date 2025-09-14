## Computational and Algorithmic Foundations
The TQE framework models the evolution of universes using a discrete-time, Monte Carlo-based simulation. The model's core elements are not based on closed-form analytical formulas but on procedural rules that govern the state evolution of each universe, guided by parameters defined in the MASTER_CTRL configuration.

Core Variables
Each universe in the simulation is characterized by three primary variables:

Energy (E): A scalar value representing the initial energy state of the universe. This parameter typically remains constant throughout the simulation.

Information (I): A normalized value in the [0, 1] interval that describes the structural complexity or disorder within the universe. Its value is derived from quantum information metrics (KL divergence and Shannon entropy).

Interaction Variable (X): A derived parameter that encapsulates the interaction between Energy and Information into a single variable (e.g., X=E⋅I). This variable determines the magnitude of the noise affecting the system's dynamics.

State Evolution and Stability
During the simulation, the parameters describing the universe's internal laws (represented as A, ns, H in the code) evolve stochastically at each time step. The stability of the system and the "locking in" of its laws depend on the magnitude of these fluctuations.

Stability: A universe is considered stable when the aggregate relative change of its internal parameters (delta_rel) remains below a predefined threshold (REL_EPS_STABLE) for a specific number of consecutive time steps (CALM_STEPS_STABLE).

Law Lock-in: This is a stricter and more permanent form of stability. The system achieves this state when the moving average of the relative change, calculated over a defined window (LOCKIN_WINDOW), falls below an even lower threshold (REL_EPS_LOCKIN) and remains there for the required duration. This represents the solidification of physical laws within that universe.

Information and Entropy
The theoretical basis for the Information (I) parameter and the entropy analysis (used in the post-analysis of the "best" universes) is Shannon Entropy. This metric quantifies the uncertainty or information content of a system.

S=− 
i
∑
​
 p 
i
​
 logp 
i
​
 
Where p 
i
​
  is the probability of the system being in the i-th microstate. Within this framework, this principle applies to the probability distributions derived from quantum states and the entropy evolution of simulated regions.
