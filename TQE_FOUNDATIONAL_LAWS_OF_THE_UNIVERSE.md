# TQE_Foundational Laws of the Universe (TQE Framing)

A structured hierarchy of canonical physical laws expressed through the energy–information coupling `f(E, I)` that underpins the Theory of the Question of Existence (TQE).

## Hierarchical Overview

| Level | Governing Law | Canonical Equation | TQE Coupling View |
| --- | --- | --- | --- |
| 0 | Master coupling | `P'(\psi) = P(\psi) · f(E, I)` | Energy–information selector for quantum outcomes |
| 1 | Conservations | `ΔE = 0`, `\vec{p}_\text{tot} = const.`, `\vec{L}_\text{tot} = const.` | `f(E, I) = 1` in isolated subsystems |
| 2 | Thermodynamics | `ΔU = Q - W`, `ΔS \ge 0`, `\lim_{T→0} S = const.` | Coupling steers probability flow toward higher entropy |
| 3 | Gravitation | `F = G m_1 m_2 / r^2`, `G_{\mu\nu} + Λ g_{\mu\nu} = (8πG/c^4) T_{\mu\nu}` | `f(E, I)` biases toward lower potentials / curved spacetime |
| 4 | Relativity | `E = mc^2`, `E^2 = (pc)^2 + (mc^2)^2` | Coupling rescales energy–mass balance |
| 5 | Electromagnetism | Maxwell equations | Coupling weights field configurations via potential energy |
| 6 | Quantum dynamics | `iħ ∂Ψ/∂t = ĤΨ`, `Δx Δp ≥ ħ/2`, `P = |Ψ|^2` | `f(E, I)` acts through the Hamiltonian on amplitudes |
| 7 | Fundamental forces | QCD + electroweak Lagrangians | Coupling embeds interaction-specific potentials |
| 8 | Cosmology | `v = H_0 d`, Friedmann equations | Global `f(E, I)` controls large-scale expansion stats |

## Level 0 – Master Coupling Law

The TQE fundamental statement is

$$
P'(\psi) = \frac{P(\psi)\, f(E, I)}{Z(E, I)}, \qquad
f(E, I) = \exp\!\left(-\frac{(E - E_c)^2}{2\sigma^2}\right) \left(1 + \alpha I \right),
$$

where `E` denotes vacuum-energy fluctuations, `I` is the information-orientation parameter (0–1), `E_c` the Goldilocks energy centre, `σ` the stability width, and `α` controls the informational influence. `Z(E, I) = ∫_Ψ P(ϕ) f(E(ϕ), I(ϕ)) dϕ` enforces probability conservation. When `f(E, I) = 1`, familiar conservation laws reappear; deviations encode statistical preference for law lock-in.

## Level 1 – Conservation Laws

- **Energy**: `ΔE = 0`. In a closed system `⟨E⟩' = ⟨E⟩`, so `f(E, I) = 1`.
- **Linear momentum**: `\vec{p}_\text{tot} = const.`; coupling remains unity without external impulses.
- **Angular momentum**: `\vec{L}_\text{tot} = const.` provided no external torque. Any non-trivial `f` would imply external information flow.

## Level 2 – Thermodynamics

- **First law**: `ΔU = Q - W`. To keep the weight positive even for large excursions, use \( f(E, I) = \exp[(Q - W)/\langle E\rangle] \), which reduces to \( 1 + (Q - W)/\langle E\rangle \) for small perturbations.
- **Second law**: `ΔS \ge 0`, with entropy \( S = -k_B \sum P(\psi) \log P(\psi) \). Here \( f(E, I) = \exp(\Delta S / k_B) \) spreads probability weight toward higher-entropy microstates.
- **Third law**: `\lim_{T→0} S = const.` implying `f(E, I) → 1`, because fluctuations freeze out as temperature approaches absolute zero.

## Level 3 – Gravitation

- **Newtonian limit**: \( F = G m_1 m_2 / r^2 \), potential \( U(r) = -G m_1 m_2 / r \). Thermal weighting yields \( f(E, I) = \exp(-U(r) / k_B T) \), i.e., deeper potentials correspond to higher weighting when coupled to a heat bath.
- **General relativity**: `G_{\mu\nu} + Λ g_{\mu\nu} = (8πG/c^4) T_{\mu\nu}`. The coupling can be viewed as an effective rescaling `T_{\mu\nu} → f(E, I) T_{\mu\nu}`, which then feeds back into the curvature through Einstein’s equations.

## Level 4 – Relativity

- **Mass–energy equivalence**: `E = mc^2`; the coupling is a constant rescaling, `f(E, I) = E / (mc^2) = 1`.
- **Relativistic dispersion**: `E^2 = (pc)^2 + (mc^2)^2`. The dimensionless form `f(E, I) = E / (mc^2) = √{1 + (p/mc)^2}` captures how momentum information perturbs rest-energy weighting.

## Level 5 – Electromagnetism

The Maxwell equations (`∇·E = ρ/ε₀`, `∇·B = 0`, `∇×E = -∂B/∂t`, `∇×B = μ₀J + μ₀ε₀ ∂E/∂t`) follow once the coupling favours field configurations consistent with charge and current distributions. For static fields one can express the bias via the electrostatic potential `φ(r)`:

$$
f(E, I) = \exp\!\left(-\frac{q\,\phi(r)}{k_B T}\right),
$$

indicating that higher potential energy states receive exponentially suppressed probability weight in thermal ensembles. Here `E` refers to the vacuum-energy background entering the master coupling, while `φ(r)` is the local potential energy density that biases charge configurations.

## Level 6 – Quantum Mechanics

- **Schrödinger equation**: `iħ ∂Ψ/∂t = Ĥ Ψ`. The Hamiltonian `Ĥ` embodies `f(E, I)`—it sets how energy and information content drive amplitude flow, but the wavefunction must be renormalised so that `⟨Ψ|Ψ⟩ = 1`.
- **Measurement postulate**: `P(ψ) = |Ψ|^2`. Deviations `f(E, I) ≠ 1` would show up as altered collapse statistics, yet these must remain consistent with unitarity.
- **Heisenberg uncertainty**: `Δx · Δp ≥ ħ/2`. Since the coupling cannot break canonical commutators, this bound is preserved even with non-trivial `f`.

## Level 7 – Fundamental Interactions

- **Strong force (QCD)**: The Cornell-like potential `V(r) ≈ -(4/3) α_s ħc / r + k r` leads to confinement-weighted coupling `f(E, I) = exp(-V(r)/k_B T)`.
- **Weak interaction**: The Fermi Lagrangian `𝓛_\text{weak} ~ G_F (\bar{ψ} γ_\mu ψ)(\bar{ψ} γ^\mu ψ)` embeds short-range information transfer; `f(E, I)` can incorporate chirality-sensitive information measures (e.g., left/right occupation imbalances) through `G_F`.

## Level 8 – Cosmology

- **Hubble law**: `v = H_0 d`. A dimensionless, positive weighting uses `f(E, I) = exp(β H_0 d)`, capturing how large-scale separations inherit the cosmic expansion bias.
- **Friedmann dynamics**:

$$
\left(\frac{\dot{a}}{a}\right)^2 = \frac{8\pi G}{3} \rho - \frac{k}{a^2} + \frac{\Lambda}{3},
$$

where `f(E, I)` modifies the effective sources through `ρ_eff = f ρ` or `Λ_eff = f Λ`, linking microscopic coupling to macroscopic expansion history.

## Cohesive Picture

The eight-level hierarchy shows that every canonical equation can be rewritten as either a neutral (`f = 1`) or weighted (`f ≠ 1`) manifestation of the same master coupling. Conservation laws arise when the coupling is dormant, thermodynamic and gauge structures when it biases probabilistic flow, and cosmological behaviour when it rescales large-scale sources. In this framing, the TQE architecture is simply the bookkeeping that tracks how `f(E, I)` turns microscopic energy–information preferences into the full catalogue of macroscopic laws.

