# TQE Cycle Model

The cyclic evolution of the Theory of the Question of Existence (TQE) expressed through a single iterative update rule, showing how the pre-fluctuation, collapse, stabilisation, heat-death, and reset regimes align with time-dependent choices of selection strength, admissible law-space, and fitness functions.

## 0) Master Law (valid for every phase)

The core rule is a normalised weighting over the space of candidate law-states `ψ`:

$$
P'(\psi) = \frac{P(\psi)\, f(E(\psi), I(\psi))}{\int_\Psi P(\phi) f(E(\phi), I(\phi))\, d\phi},
$$

with a typical parametric form

$$
f(E, I) = \exp[\beta(t) X(\psi) - \lambda_{\text{out}}(t)], \qquad X(\psi) \in G_t / t.
$$

- `β(t)` sets the selection pressure (analogous to inverse temperature).
- `G_t` is the admissible “gate” region in law-space.
- `λ_out(t)` ensures normalisation and enforces hard constraints.

### Iterative update

$$
P_{k+1}(\psi) = P_k(\psi)\, f_k(E(\psi), I(\psi)) = P_k(\psi)\, \exp[\beta_k X_k(\psi)],
$$

renormalised via

$$
\tilde{P}_{k+1}(\psi) = \frac{P_{k+1}(\psi)}{\int_\Psi P_{k+1}(\phi)\, d\phi}.
$$

In the continuum limit, this becomes a replicator equation

$$
\partial_t P_t(\psi) = P_t(\psi) \left[G_t(\psi) - \mathbb{E}_t(G_t)\right],
$$

where `G_t` encapsulates the instantaneous gain (or fitness) derived from the energy–information coupling.

### Explicit β(t) prescriptions

To close the dynamics, β(t) can be supplied analytically:

- **Logistic ramp** (slow turn-on around `t_c`):

  $$
  \beta(t) = \frac{\beta_{\max}}{1 + e^{-k (t - t_c)}}.
  $$

  This captures the gradual strengthening of selection leading into the lock-in epoch.

- **Thermal inverse** (coupled to cosmic expansion):

  $$
  \beta(t) = \frac{1}{k_B T(t)}, \qquad T(t) = T_0 \left[\frac{a(t)}{a_0}\right]^{-n},
  $$

  so that larger scale factors dilute the effective temperature and raise β.

- **Relaxation + noise**:

  $$
  \dot{\beta}(t) = \kappa \left[\beta_{\text{eq}} - \beta(t)\right] + \xi(t),
  $$

  with `κ` the relaxation constant and `ξ(t)` a stochastic fluctuation term.

## A) t < 0 — Fluctuation + Superposition (Pre-selection)

The cosmic state is a high-variance superposition with weak selection (`β(t) ≈ 0`). Two regimes are important:

- **Soft collapse (large β, Gibbs weighting):**

  $$
  P_0^+(\psi) = \frac{P_0^-(\psi)\, e^{-\beta_0 X(\psi)}}{\int P_0^-(\phi) e^{-\beta_0 X(\phi)} d\phi}.
  $$

- **Hard collapse (projector onto gate G):**

  $$
  P_0^+(\psi) = \frac{P_0^-(\psi)\, \mathbf{1}_{X(\psi)\in G}}{\int P_0^-(\phi)\, \mathbf{1}_{X(\phi)\in G} \, d\phi}.
  $$

Here, `G` defines the nascent law-space (gauge class, Lorentz sector, GR-compatible states, etc.).

## B) t = 0 — Collapse (Initial Condition Fixing)

Quantum fluctuations yield a still-disordered but sharply localised state. The probability distribution narrows, yet full law lock-in has not occurred. Stable regions in the state space start to emerge. `β(t)` ramps up, shrinking the accessible phase space.

## C) t > 0 — Expansion → Stabilisation → Complexity (Selection + Exploration)

Post-collapse, the replicator dynamics continue with increasing selection. During numerical experiments this manifests around step ~300–310, where symmetry-breaking pressure crystallises:

- **Conservation laws**: time, spatial, and rotational invariance induce energy, momentum, and angular momentum conservation.
- **Quantum regime**: `f ~ e^{iS/ħ}` yields Schrödinger evolution, Born rule, and Heisenberg relations.
- **Gauge symmetries**: `U(1)`, `SU(2) × U(1)`, `SU(3)` engender electromagnetism, weak, and strong interactions.
- **General relativity + cosmology**: diffeomorphism invariance enforces Einstein’s equations; FLRW symmetry gives Friedmann dynamics.

## D) Late Times — March Toward Heat Death

As vacuum energy dominates (`a(t) ~ e^{H_Λ t}`), the de Sitter temperature `T_dS = ħ H_Λ / (2π k_B)` becomes the governing thermal scale.

- **Black hole evaporation**: `dM/dt = -α ħ c^4 / (G^2 M^2)` drives masses toward zero, with total evaporation time `t_evap ∝ G^2 M^3 / (α ħ c^4)`.
- **Entropy monotonicity**: `S_tot(t) → S_∞`, `Var[X_t] → max`, `β(t) → 0`. Chemical potentials vanish (`μ → 0`), freezing useful free energy.

## E) “Clean Slate” — Speculative Rebirth (Return to t < 0)

When `β(t) → 0` and the variance of the fitness landscape reaches its maximum—i.e., the selection force disappears—the locked-in laws dissolve. Possible trigger mechanisms include:

- Coleman–De Luccia vacuum bubble nucleation (`Γ/V ~ A e^{-B/ħ}`),
- Loop quantum cosmology bounce,
- Conformal cyclic cosmology matching.

A formal reset map `Π` reparametrises the distribution:

$$
P_{\text{new},0^-}(\psi) := \Pi\!\left[\lim_{t \to \infty} P_t(\psi)\right],
$$

effectively dismantling the previous `G` and reopening the high-variance exploration regime with small β. The cycle restarts.

### Candidate Π operators

- **Entropic re-sampling**:

  $$
  \Pi[P](\psi) = \frac{e^{-\gamma S[P]} P(\psi)}{\int e^{-\gamma S[P]} P(\phi)\, d\phi},
  $$

  with `S[P] = -∫ P log P dψ` and `γ` governing how much information is retained between cycles.

- **Perturbative reset**:

  $$
  \Pi[P](\psi) = P(\psi) + \varepsilon\, \eta(\psi),
  $$

  where `η(ψ)` is a zero-mean random field and `ε ≪ 1`, injecting fresh fluctuations.

- **Cyclic rescaling**:

  $$
  \Pi[P](\psi) = \frac{P(\psi/\alpha)}{\alpha},
  $$

  which rescales the state variable by `α` to represent geometric contraction/expansion at the start of a new aeon.

### Numerical stability considerations

The discrete update

$$
P_{k+1}(\psi) = P_k(\psi) \exp[\beta_k X_k(\psi)], \qquad
\tilde{P}_{k+1}(\psi) = \frac{P_{k+1}(\psi)}{\int P_{k+1}(\phi) d\phi}
$$

admits several diagnostics:

1. **Linear stability**: perturbations obey

   $$
   \delta P_{k+1}(\psi) \approx \left[1 + \beta_k X'_k(\psi)\right] \delta P_k(\psi),
   $$

   so convergence requires `|1 + β_k X'_k| < 1` over the support of `P_k`.

2. **Monte Carlo verification**: iterate the map for representative β-k schedules and fitness landscapes `X_k(ψ)` to chart regimes of convergence, oscillation, or chaos.

3. **Continuum-limit check**: numerically demonstrate that the discrete scheme approaches the replicator PDE

   $$
   \partial_t P_t(\psi) = P_t(\psi) \left[G_t(\psi) - \mathbb{E}_t(G_t)\right]
   $$

   under appropriate scaling of `Δt`, `β_k`, and `X_k`.

## Integrated Cycle View

Taken together, the β-schedules, Π operators, and stability diagnostics turn the qualitative story into a closed dynamical loop:

1. **Exploration** (β low, high-variance P_t).  
2. **Collapse + Lock-in** (β ramps via logistic/thermal law, enforcing gate `G_t`).  
3. **Stabilisation** (replicator flow preserves conserved symmetries).  
4. **Heat death** (β → 0, entropy saturates, Π prepares reset).  
5. **Reset** (Π[P] re-seeds the ensemble, returning to step 1).

This cycle provides a mathematically specified scaffold for the broader TQE narrative: physical laws emerge, persist, and eventually relax through repeated application of the master coupling and its reset mechanics.

