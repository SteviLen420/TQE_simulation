# TQE Cycle Model

A complete specification of the Theory of the Question of Existence (TQE) dynamics, showing how the master coupling, time-dependent selection strength, gate structure, and reset operator orchestrate the progression from pre-law superposition to law lock-in, heat death, and rebirth. This revision emphasises the *story* told by the equations: how a single bias function $f(E,I)$ repeatedly sculpts a universe from vacuum noise, freezes its laws, lets it drift toward heat death, and finally resets the stage. Think of each section below as a lab notebook entry from a PhD investigator tracking which part of the cycle is currently active and what physical intuition accompanies the formalism.

---

## 1. Master Modulation Architecture

The universe ensemble is described by a probability density $P(\psi)$ over candidate law-states $\psi$. Every phase of the cycle applies the same modulation law
$$
P'(\psi) = \frac{P(\psi)\, f_{\text{gate}}(\psi, t)}{\int_\Psi P(\phi)\, f_{\text{gate}}(\phi, t)\, d\phi},
$$
with
$$
f_{\text{gate}}(\psi, t) = \exp\!\big[\beta(t) X(\psi)\big]\, \mathbf{1}_{\psi \in G_t} \, e^{-\lambda_{\text{out}}(t)}.
$$
Here the indicator projects out configurations that violate the gate, leaving a closed, positive operator on the admissible subspace. The Lagrange multiplier is tied to explicit constraints $\mathcal{C}_j[P]=0$ (normalisation, symmetry closure, total energy bounds, etc.) via $\lambda_{\text{out}}(t) = \sum_j \Lambda_j(t)\, \mathcal{C}_j'[P_t]$ so that the same functional form encodes both probability conservation and hard structural rules.

- $\beta(t)$ — selection pressure (inverse temperature analog).  
- $X(\psi)$ — fitness/complexity functional.  
- $G_t$ — admissible gate in law-space (enforcing symmetries, integrability, etc.).  
- $\lambda_{\text{out}}(t)$ — Lagrange multiplier adjusting for hard constraints.

### Discrete update vs. continuum limit

For simulations:

- **Update** 

$$
P_{k+1}(\psi) = P_k(\psi)\, \exp\big[\beta_k X_k(\psi)\big]\, \mathbf{1}_{\psi \in G_{t_k}}\, e^{-\lambda_{\text{out},k}}.
$$

- **Renormalise** 

$$
\tilde{P}_{k+1}(\psi) = \frac{P_{k+1}(\psi)}{\int_\Psi P_{k+1}(\phi)\, d\phi}.
$$

As $\Delta t \to 0$, the continuum limit obeys
$$
\partial_t P_t(\psi) = P_t(\psi)\, [G_t(\psi) - \mathbb{E}_t(G_t)] - \kappa_t(\psi),
$$
where $G_t(\psi) \equiv \beta(t) X(\psi)$ and $\kappa_t(\psi)$ enforces the gate projection (either as an absorbing boundary term or as a constraint ensuring $P_t(\psi)=0$ outside $G_t$). This is the standard replicator equation with payoff given directly by the energy–information coupling.

**Interpretation.** This is the backbone of TQE: no matter which era we examine, reality updates its law-distribution by multiplying with the same exponential bias and renormalising. The selection pressure $\beta(t)$ acts like an inverse temperature dial, the gate $G_t$ is a moving boundary that decides which candidate laws are even admissible, and the Lagrange multiplier enforces hard constraints (normalisation, gauge closure, etc.). In practice, each cosmological phase simply corresponds to a different choice of these schedules. The replicator equation makes the statistical analogy explicit: physical law behaves like a population whose “fitness” is short-term stability plus long-term complexity support.

---

## 2. Time-Dependent Selection Strength $\beta(t)$

The cycle is driven by how quickly selection pressure rises, saturates, and falls. Useful closed forms:

1. **Logistic ramp** — $\beta(t) = \frac{\beta_{\max}}{1 + e^{-k (t - t_c)}}$ for smooth onset near $t_c$.

2. **Thermal inverse tied to expansion** — $\beta(t) = 1/[k_B T(t)]$ with $T(t) = T_0 [a(t)/a_0]^{-n}$ so cosmological cooling naturally increases selection.

3. **Relaxation with stochastic forcing** — $\dot{\beta}(t) = \kappa [\beta_{\text{eq}} - \beta(t)] + \xi(t)$, with relaxation constant $\kappa$ and mean-zero noise $\xi(t)$ interpreted in the Itô sense (so $\mathrm{d}\beta = \kappa [\beta_{\text{eq}} - \beta]\, \mathrm{d}t + \sigma_\beta\, \mathrm{d}W_t$ and any noise-induced drift is explicit). Stratonovich variants can be adopted but must then include the appropriate correction term.

**Interpretation.** $\beta(t)$ is the “metronome” of the entire model. When it is near zero, the universe behaves like a hot, information-neutral plasma in law-space—everything is allowed, nothing is preferred. As $\beta$ ramps up, the Goldilocks window narrows and only configurations with the right energy–information balance remain statistically relevant. The specific functional form matters physically: a logistic ramp corresponds to a smooth reheating or symmetry-breaking event; the inverse-temperature law ties selection directly to cosmological cooling; the stochastic relaxation captures messy epochs where structure-formation or black-hole evaporation inject noise. Keeping $\beta(t)$ explicit is what gives TQE predictive leverage over when lock-in occurs.

---

## 3. Gate Structure and Fitness Functionals

The gate $G_t$ partitions law-space into allowed and forbidden regions, e.g., enforcing Lorentz invariance, gauge algebra closure, or diffeomorphism symmetry. Fitness functionals are taken as explicit composites
$$
X(\psi) = \alpha_1 C_{\text{sym}}(\psi) + \alpha_2 C_{\text{stab}}(\psi) + \alpha_3 C_{\text{complex}}(\psi),
$$
where each $C_i : \Psi \to \mathbb{R}$ is a measurable score (symmetry residual norm, linear-stability margin, information-throughput metric, etc.) and $\alpha_i$ fix the trade-offs. Energy–information composites, conservation compliance, and anomaly penalties are special cases of this decomposition.

The master coupling only acts inside $G_t$; violations are projected out by setting $f = 0$ there.

**Interpretation.** Gates are the rulebook. Without them, the exponential bias would happily amplify unphysical states. By enforcing $G_t$ before weighting, we make sure that only law-sets capable of supporting coherent dynamics are even considered. In simulations this is literally a mask over configuration space; conceptually it represents deep structural facts (symmetry groups, anomaly cancellation, integrability) that must hold if observers are ever to appear. The fitness functional $X(\psi)$ then scores how well a candidate law supports complexity once it passes the gate. Together, $(G_t, X)$ encode the “selection environment” for an epoch.

---

## 4. Phase Anatomy

This section follows a single patch of the cycle and narrates what the modulation law is doing at each stage. Rather than new formulas, we provide the physical storyline connecting the raw equations to cosmological intuition.

### 4.1 Pre-selection ($t < 0$)

High-variance superposition with $\beta(t) \approx 0$. Two equivalent initialisations:

- **Soft (Gibbs) collapse** 

$$
P_0^+(\psi) = \frac{P_0^-(\psi)\, e^{-\beta_0 X(\psi)}}{\int P_0^-(\phi)\, e^{-\beta_0 X(\phi)}\, d\phi}.
$$

- **Hard projection** 

$$
P_0^+(\psi) = \frac{P_0^-(\psi)\, \mathbf{1}_{X(\psi)\in G}}{\int P_0^-(\phi)\, \mathbf{1}_{X(\phi)\in G}\, d\phi}.
$$

These represent, respectively, probabilistic biasing and strict gate enforcement before a universe crystallises.

**Interpretation.** Pre-selection is pure exploration. The ensemble has almost no orientation; the KL divergence between updates is tiny, so $f(E,I) \approx 1$ everywhere. The “soft” variant is the statistical mechanic’s approach—nudge probabilities gently according to small biases—while the “hard” projection models theories where only symmetry-respecting states survive the first cut. In either case, the system remains lawless but poised for collapse.

### 4.2 Collapse point ($t = 0$)

Vacuum fluctuations spontaneously localise $P(\psi)$ but do not yet select a unique law set. $\beta(t)$ begins to increase and the accessible region in configuration space shrinks.

**Interpretation.** The collapse point is the TQE analogue of a phase transition: a single draw from the vacuum sets the initial complexity parameter $X = E \cdot I$. The distribution suddenly narrows, yet multiple law candidates still compete. Observationally this would correspond to the end of the primordial fluctuation epoch where the universe gains a definite energy budget but not yet fixed couplings.

### 4.3 Growth, stabilisation, complexity ($t > 0$)

- Conservation laws emerge when the replicator fixed points $P^\star(\psi)$ satisfy $P^\star(\psi)=0$ unless $\psi \in \arg\max_{\phi \in G_t} X(\phi)$, i.e., only the top-scoring gate-compliant laws retain weight.  
- Quantum behaviour requires extending the bias into a two-channel evolution: $f_{\text{amp}} = \exp[\beta X]$ for amplitudes and a separate phase channel $\exp[i S(\psi)/\hbar]$ so that the effective dynamics remain unitary while the selection bias acts on moduli.  
- Gauge groups $U(1)$, $SU(2)\times U(1)$, $SU(3)$ dominate where $G_t$ admits them, leading to electromagnetic, weak, and strong sectors.  
- GR + cosmology follow once diffeomorphism invariance becomes energetically favoured, producing Einstein and Friedmann dynamics.

**Interpretation.** This is the productive era: $\beta(t)$ has climbed high enough that only a narrow Goldilocks zone survives, so the replicator dynamics settle into attractors. Those attractors are the familiar laws—conservation, QM, gauge theory, GR—each appearing as the most information-efficient way to keep the universe inside the window. The timeline also clarifies why some symmetries break later than others: they wait until the gate structure and fitness functional make them the winning strategy.

### 4.4 Heat-death drift (late $t$)

As shown in the black-hole cleanup analysis, vacuum energy takes over, black holes evaporate, $\beta(t) \to 0$, and variability $\mathrm{Var}[X_t]$ saturates. The system tends toward a homogeneous mixture at $T_{\text{dS}}$.

**Interpretation.** Heat death is simply the reverse of the growth era. Expansion plus horizon entropy forces $\beta$ back down, flattening $P(\psi)$. Once variation in $X$ stops decreasing, the universe is effectively law-still but dynamically empty—no more structure emerges because the bias that maintained it has faded.

### 4.5 Reset initiation

Once selection has effectively switched off, one of several triggers (CDL bubble, CCC matching, LQC bounce, or intrinsic Π-reset) restarts the cycle.

**Interpretation.** The reset is not an ad hoc button; it is the statistical necessity that once $\beta \approx 0$ the system must either remain frozen forever or be jolted back into exploration. TQE encodes that jolt via explicit Π maps (next section) so that the same master coupling can immediately begin another round. Physical candidates—bubble nucleation, conformal gluing between aeons, loop-quantum bounces—are different realizations of the same requirement: inject enough informational variance to give $f(E,I)$ something to work with again.

---

## 5. Π Operators – Formal Reset Mechanisms

The reset map rewrites the terminal distribution into a fresh high-variance state via $P_{\text{new},0^-}(\psi) = \Pi[P_\infty(\psi)]$.

Representative choices:

1. **Entropic reweighting** — $\Pi[P] (\psi) = \frac{e^{-\gamma s(\psi)} P(\psi)}{\int e^{-\gamma s(\phi)} P(\phi)\, d\phi}$ with $s(\psi)$ a local entropy-density estimator (e.g., neighbourhood entropy or KL score) so the reweighting truly reshapes the distribution.

2. **Perturbative noise injection** — $\Pi[P] (\psi) = P(\psi) + \varepsilon\, \eta(\psi)$ where $\eta$ is a zero-mean random field.

3. **Cyclic rescaling** — $\Pi[P] (\psi) = P(\psi/\alpha)/\alpha$ to capture geometric contraction/expansion at aeon boundaries.

Each option preserves normalisation and reintroduces exploratory variance before the next $\beta$ ramp.

**Interpretation.** Π is the mathematical placeholder for “whatever physics clears the board.” The entropic option models a universe where holographic horizon data is recycled with a tunable bias $\gamma$. Noise injection captures quantum-gravity inspired randomness. Cyclic rescaling matches conformal or ekpyrotic ideas. What matters is not the micro-details but that Π increases variance while respecting normalisation, ensuring the next cycle begins from a true pre-selection ensemble instead of a partially frozen relic.

---

## 6. Numerical Stability and Verification

Because $P_{k+1}(\psi) = P_k(\psi)\, e^{\beta_k X_k(\psi)}$ multiplies densities, stability conditions are essential:

1. **Linearised perturbations**

   $\delta P_{k+1}(\psi) \approx [1 + \beta_k X'_k(\psi)]\, \delta P_k(\psi)$ under an explicit-Euler interpretation of the discrete map, so $|1 + \beta_k X'_k| < 1$ ensures local convergence.

2. **Monte Carlo sweeps** over $\beta_k$ schedules and fitness landscapes map out convergence vs. oscillatory vs. chaotic regimes; chaos is diagnosed by a positive Lyapunov exponent $\lambda = \lim_{n\to\infty} \frac{1}{n} \ln \frac{\|\delta P_n\|}{\|\delta P_0\|}$.

3. **Continuum limit**: verify numerically that the discrete map converges to the replicator PDE when $\beta_k \propto \Delta t$ and $X_k$ varies smoothly.

**Interpretation.** These diagnostics translate the cycle from philosophy to code. Linearised perturbations warn us when selection pressure is too aggressive and would cause oscillations instead of convergence. Monte Carlo sweeps reveal whether a planned $\beta$ schedule pushes the system into chaotic regimes (useful for reset studies) or stable ones (useful for lock-in). The continuum check ensures that the discrete simulations we run are faithful representations of the analytic replicator dynamics. Together they supply the “engineering tolerances” of the theory.

---

## 7. Integrated Cycle Summary

1. **Exploration:** $\beta \approx 0$, high-variance $P$, broad gate.  
2. **Collapse and lock-in:** $\beta(t)$ ramps via logistic/thermal law, enforcing $G_t$ and privileging symmetry-respecting states.  
3. **Stabilisation:** The replicator fixed points manifest as conservation laws, quantum rules, gauge structures, and GR.  
4. **Heat death:** Black-hole evaporation and $\Lambda$-domination drive $\beta \to 0$, storing residual information on horizons.  
5. **Reset:** Π acts (possibly assisted by CDL/CCC/LQC mechanisms) to seed the next pre-selection phase.

The reset trigger is implemented operationally by monitoring (i) variance thresholds $\mathrm{Var}[X_t] < \varepsilon_X$, (ii) entropy plateaus $S[P_t] > S_{\text{max}} - \varepsilon_S$, and (iii) gate softening parameters $\gamma_t \to \gamma_{\text{soft}}$; once all three conditions hold for a dwell time $\tau_{\text{reset}}$, the Π operator is applied and $\beta$ schedules restart at their low-variance initial values.

This closed loop is the operational backbone of TQE: a single modulation rule, equipped with explicit $\beta$-schedules, gate logic, reset operators, and stability diagnostics, suffices to describe how physical law can repeatedly emerge, persist, and dissolve across successive cosmological epochs.

**Interpretation.** At a narrative level, the cycle says: orientational bias wakes up, assembles a universe, quietly winds down, and hands the informational residue to the next iteration. Because every stage reuses the same $P' = P f / Z$ machinery, the theory remains falsifiable—alter a piece (say, the $\beta$ ramp) and the downstream predictions for lock-in times, anomaly statistics, or reheating signatures change accordingly. The cycle model is therefore both a conceptual map and a simulation-ready recipe.
