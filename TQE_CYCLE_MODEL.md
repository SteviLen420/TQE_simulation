# TQE Cycle Model

A complete specification of the Theory of the Question of Existence (TQE) dynamics, showing how the master coupling, time-dependent selection strength, gate structure, and reset operator orchestrate the progression from pre-law superposition to law lock-in, heat death, and rebirth.

---

## 1. Master Modulation Architecture

The universe ensemble is described by a probability density $P(\psi)$ over candidate law-states $\psi$. Every phase of the cycle applies the same modulation law $P'(\psi) = \frac{P(\psi)\, f(E(\psi), I(\psi))}{\int_\Psi P(\phi)$ , $f(E(\phi), I(\phi))\, d\phi}$ with fine-tuning factor $f(E, I) = \exp[\beta(t) X(\psi) - \lambda_{\text{out}}(t)]$ restricted to $X(\psi) \in G_t / t$.

- $\beta(t)$ — selection pressure (inverse temperature analog).  
- $X(\psi)$ — fitness/complexity functional.  
- $G_t$ — admissible gate in law-space (enforcing symmetries, integrability, etc.).  
- $\lambda_{\text{out}}(t)$ — Lagrange multiplier adjusting for hard constraints.

### Discrete update vs. continuum limit

For simulations, use $(P_{k+1}(\psi) = P_k(\psi)\, \exp[\beta_k X_k(\psi)])$ and renormalise with $(\tilde{P}_{k+1}(\psi) = \frac{P_{k+1}(\psi)}{\int_\Psi P_{k+1}(\phi)\, d\phi})$. As $\Delta t \to 0$, the continuum limit obeys $\partial_t P_t(\psi) = P_t(\psi)\, [G_t(\psi) - \mathbb{E}_t(G_t)]$, the standard replicator equation with “game payoff” $G_t(\psi)$ set by the energy–information coupling.

---

## 2. Time-Dependent Selection Strength $\beta(t)$

The cycle is driven by how quickly selection pressure rises, saturates, and falls. Useful closed forms:

1. **Logistic ramp** — $\beta(t) = \frac{\beta_{\max}}{1 + e^{-k (t - t_c)}}$ for smooth onset near $t_c$.

2. **Thermal inverse tied to expansion** — $\beta(t) = 1/[k_B T(t)]$ with $T(t) = T_0 [a(t)/a_0]^{-n}$ so cosmological cooling naturally increases selection.

3. **Relaxation with stochastic forcing** — $\dot{\beta}(t) = \kappa [\beta_{\text{eq}} - \beta(t)] + \xi(t)$, with relaxation constant $\kappa$ and mean-zero noise $\xi(t)$.

---

## 3. Gate Structure and Fitness Functionals

The gate $G_t$ partitions law-space into allowed and forbidden regions, e.g., enforcing Lorentz invariance, gauge algebra closure, or diffeomorphism symmetry. Fitness functionals $X(\psi)$ can be:

- energy–information composites $X = g(E,I)$,  
- conservation compliance scores,  
- anomaly detectors penalizing unstable configurations.

The master coupling only acts inside $G_t$; violations are projected out by setting $f = 0$ there.

---

## 4. Phase Anatomy

### 4.1 Pre-selection ($t < 0$)

High-variance superposition with $\beta(t) \approx 0$. Two equivalent initialisations:

- **Soft (Gibbs) collapse** — $P_0^+(\psi) = \frac{P_0^-(\psi)\, e^{-\beta_0 X(\psi)}}{\int P_0^-(\phi)\, e^{-\beta_0 X(\phi)}\, d\phi}$.

- **Hard projection** — $P_0^+(\psi) = \frac{P_0^-(\psi)\, \mathbf{1}_{X(\psi)\in G}}{\int P_0^-(\phi)\, \mathbf{1}_{X(\phi)\in G}\, d\phi}$.

These represent, respectively, probabilistic biasing and strict gate enforcement before a universe crystallises.

### 4.2 Collapse point ($t = 0$)

Vacuum fluctuations spontaneously localise $P(\psi)$ but do not yet select a unique law set. $\beta(t)$ begins to increase and the accessible region in configuration space shrinks.

### 4.3 Growth, stabilisation, complexity ($t > 0$)

- Conservation laws emerge when the replicator fixed points respect time, space, and rotational invariance.  
- Quantum behaviour corresponds to the unitary sector $f \sim e^{iS/\hbar}$.  
- Gauge groups $U(1)$, $SU(2)\times U(1)$, $SU(3)$ dominate where $G_t$ admits them, leading to electromagnetic, weak, and strong sectors.  
- GR + cosmology follow once diffeomorphism invariance becomes energetically favoured, producing Einstein and Friedmann dynamics.

### 4.4 Heat-death drift (late $t$)

As shown in the black-hole cleanup analysis, vacuum energy takes over, black holes evaporate, $\beta(t) \to 0$, and variability $\mathrm{Var}[X_t]$ saturates. The system tends toward a homogeneous mixture at $T_{\text{dS}}$.

### 4.5 Reset initiation

Once selection has effectively switched off, one of several triggers (CDL bubble, CCC matching, LQC bounce, or intrinsic Π-reset) restarts the cycle.

---

## 5. Π Operators – Formal Reset Mechanisms

The reset map rewrites the terminal distribution into a fresh high-variance state via $P_{\text{new},0^-}(\psi) = \Pi[P_\infty(\psi)]$.

Representative choices:

1. **Entropic reweighting** — $\Pi[P](\psi) = \frac{e^{-\gamma S[P]} P(\psi)}{\int e^{-\gamma S[P]} P(\phi)\, d\phi}$ with $S[P] = -\int P \ln P\, d\psi$.

2. **Perturbative noise injection** — $\Pi[P](\psi) = P(\psi) + \varepsilon\, \eta(\psi)$ where $\eta$ is a zero-mean random field.

3. **Cyclic rescaling** — $\Pi[P](\psi) = P(\psi/\alpha)/\alpha$ to capture geometric contraction/expansion at aeon boundaries.

Each option preserves normalisation and reintroduces exploratory variance before the next $\beta$ ramp.

---

## 6. Numerical Stability and Verification

Because $P_{k+1}(\psi) = P_k(\psi)\, e^{\beta_k X_k(\psi)}$ multiplies densities, stability conditions are essential:

1. **Linearised perturbations**

   $\delta P_{k+1}(\psi) \approx [1 + \beta_k X'_k(\psi)]\, \delta P_k(\psi)$, so $|1 + \beta_k X'_k| < 1$ ensures local convergence.

2. **Monte Carlo sweeps** over $\beta_k$ schedules and fitness landscapes map out convergence vs. oscillatory vs. chaotic regimes.

3. **Continuum limit**: verify numerically that the discrete map converges to the replicator PDE when $\beta_k \propto \Delta t$ and $X_k$ varies smoothly.

---

## 7. Integrated Cycle Summary

1. **Exploration:** $\beta \approx 0$, high-variance $P$, broad gate.  
2. **Collapse and lock-in:** $\beta(t)$ ramps via logistic/thermal law, enforcing $G_t$ and privileging symmetry-respecting states.  
3. **Stabilisation:** The replicator fixed points manifest as conservation laws, quantum rules, gauge structures, and GR.  
4. **Heat death:** Black-hole evaporation and Λ-domination drive $\beta \to 0$, storing residual information on horizons.  
5. **Reset:** Π acts (possibly assisted by CDL/CCC/LQC mechanisms) to seed the next pre-selection phase.

This closed loop is the operational backbone of TQE: a single modulation rule, equipped with explicit $\beta$-schedules, gate logic, reset operators, and stability diagnostics, suffices to describe how physical law can repeatedly emerge, persist, and dissolve across successive cosmological epochs.

