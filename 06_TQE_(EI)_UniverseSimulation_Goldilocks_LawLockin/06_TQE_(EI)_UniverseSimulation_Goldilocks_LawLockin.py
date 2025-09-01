# ===========================================================================
# Theory of the Question of Existence (TQE)
# (E, I) Vacuum fluctuation → Superposition → Collapse → Expansion → Law lock-in
# ===========================================================================
# Author: Stefan Len
# Description: Full Monte Carlo pipeline starting from many-universe code
# Focus: Stable, law-locked universes via Energy–Information dynamics
# Includes: MC, law_lock_in, averaged c(t), CSV/PNG saves, summary.json, SHAP/LIME
# ===========================================================================

# ---- Mount Google Drive ----
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# ---- Minimal auto-install (Colab) ----
import sys, subprocess, warnings
warnings.filterwarnings("ignore")

def _ensure(pkg):
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

for pkg in ["qutip", "pandas", "scikit-learn", "shap", "lime"]:
    _ensure(pkg)

# ---- Imports ----
import os, time, json, numpy as np, pandas as pd, matplotlib.pyplot as plt
import qutip as qt
import shap
from lime.lime_tabular import LimeTabularExplainer

# ---- Directories ----
GOOGLE_BASE = "/content/drive/MyDrive/TQE_(E,I)_UNIVERSE_SIMULATION"
run_id = time.strftime("TQE_(E,I)_UNIVERSE_SIMULATION_%Y%m%d_%H%M%S")
SAVE_DIR = os.path.join(GOOGLE_BASE, run_id); os.makedirs(SAVE_DIR, exist_ok=True)
FIG_DIR  = os.path.join(SAVE_DIR, "figs"); os.makedirs(FIG_DIR, exist_ok=True)

def savefig(p):
    plt.savefig(p, dpi=180, bbox_inches="tight")
    plt.close()

# ===== Master flags (tune as you like) =====
PLOT_AVG_LOCKIN  = False   # average lock-in c(t) plot toggle
PLOT_LOCKIN_HIST = False   # histogram of lock-in epochs plot toggle
RUN_XAI          = True    # SHAP + LIME
RUN_SEED_SEARCH  = False   # heavy; enable when needed

# ======================================================
# MASTER CONTROLLER – one source of truth for all params
# ======================================================
MASTER_CTRL = {
    # --- Core sizes ---
    "NUM_UNIVERSES": 5000,
    "TIME_STEPS": 500,
    "LOCKIN_EPOCHS": 500,
    "EXPANSION_EPOCHS": 500,

    # --- Energy distribution (lognormal) ---
    "E_LOG_MU": 2.5,
    "E_LOG_SIGMA": 0.8,

    # --- Goldilocks window (linear E) + info weight ---
    "E_CENTER": 2.0,      # was E_C
    "E_WIDTH": 0.5,       # was SIGMA
    "ALPHA_I": 0.8,       # was ALPHA

    # --- Stability thresholds ---
    "F_GATE_STABLE": 0.20,
    "F_GATE_LOCKIN": 0.10,
    "CALM_STEPS_STABLE": 5,
    "CALM_STEPS_LOCKIN": 5,
    "REL_EPS_STABLE": 0.05,   # relative change threshold for stability
    "REL_EPS_LOCKIN": 1e-3,   # relative change threshold for lock-in

    # --- Expansion dynamics coupling ---
    "EXP_GROWTH_BASE": 1.005,
    "EXP_NOISE_BASE": 1.0,
    "EXP_COUPLE_TO_X": True,   # couple growth/noise to X = E*I*f

    # --- XAI / plots / extras ---
    "RUN_XAI": True,
    "RUN_SEED_SEARCH": False,
    "PLOT_AVG_LOCKIN": False,
    "PLOT_LOCKIN_HIST": False,
    "RUN_BEST_UNIVERSE": True,  # entropy deep-dive

    # --- Optional pin set (avoid NumPy/SciPy binary issues) ---
    "USE_PINNED_ENV": False,
    "PINNED": {
        "numpy": "1.26.4",
        "scipy": "1.11.4",
        "qutip": "5.0.3",
        "scikit-learn": "1.3.2",
        "shap": "0.43.0",
        "lime": "0.2.0.1",
        "pandas": "2.2.2"
    }
}

# ======================================================
# 1) t < 0 : Quantum superposition (vacuum fluctuation)
# ======================================================

# --- Per-block RNG setup (reproducible with master_rng) ---
sub_seed_super = int(master_rng.integers(0, 2**32 - 1))
rng_super = np.random.default_rng(sub_seed_super)

# Important: QuTiP's rand_ket uses NumPy's global RNG; keep it in sync
np.random.seed(sub_seed_super)

Nlev = 12
a = qt.destroy(Nlev)

# Perturbed Hamiltonian with small random noise (use local RNG)
H0 = a.dag()*a + 0.05 * (rng_super.normal() * a + rng_super.normal() * a.dag())

# Initial state: random superposition (driven by the synced global RNG above)
psi0 = qt.rand_ket(Nlev)
rho0 = psi0 * psi0.dag()

# Coarse time grid and time-varying dissipation rate gamma(t)
tlist = np.linspace(0, 10, 200)
gammas = 0.02 + 0.01 * np.sin(0.5 * tlist) + 0.005 * rng_super.normal(size=len(tlist))

states = []
for g in gammas:
    # Short window evolution; ensure non-negative rate with abs()
    res = qt.mesolve(
        H0,
        rho0,
        np.linspace(0, 0.5, 5),
        [np.sqrt(abs(g)) * a],
        [],
        progress_bar=None  # avoid overhead in tight loops
    )
    states.append(res.states[-1])

def purity(r):
    # Works for both density matrices and kets
    return float((r * r).tr().real) if qt.isoper(r) else float((r * r.dag()).tr().real)

# Von Neumann entropy and purity traces
S = np.array([qt.entropy_vn(r) for r in states], dtype=float)
P = np.array([purity(r)       for r in states], dtype=float)

# Plot & save
plt.figure()
plt.plot(tlist, S, label="Entropy")
plt.plot(tlist, P, label="Purity")
plt.title("t < 0 : Quantum superposition (vacuum fluctuation)")
plt.xlabel("time"); plt.legend()
savefig(os.path.join(FIG_DIR, "superposition.png"))

pd.DataFrame({"time": tlist, "Entropy": S, "Purity": P}).to_csv(
    os.path.join(SAVE_DIR, "superposition.csv"),
    index=False
)

# ======================================================
# 2) t = 0 : Collapse (E·I coupling + Goldilocks factor)
# ======================================================

# KL divergence helper (safe)
def KL(p, q, eps=1e-12):
    # Clip and renormalize to avoid division by zero / log of zero
    p = np.clip(p, eps, None); q = np.clip(q, eps, None)
    p = p / p.sum(); q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))

# Information parameter I (KL × Shannon, squashed to [0,1])
def info_param(dim=8):
    """
    Uses QuTiP's rand_ket (global NumPy RNG). Make sure to sync np.random.seed()
    BEFORE calling this function if you need reproducibility.
    """
    psi1, psi2 = qt.rand_ket(dim), qt.rand_ket(dim)
    p1 = np.abs(psi1.full().ravel())**2
    p2 = np.abs(psi2.full().ravel())**2
    # Use the shared KL helper for consistency
    KL_val = KL(p1, p2)
    I_kl   = KL_val / (1.0 + KL_val)
    # Normalized Shannon entropy on p1
    eps = 1e-12
    H    = -np.sum(p1 * np.log(p1 + eps))
    I_sh = H / np.log(len(p1))
    # Multiplicative fusion, then squash back to [0,1]
    I_raw = I_kl * I_sh
    return I_raw / (1.0 + I_raw)

# Energy sampling with optional RNG
def sample_energy(mu=2.5, sigma=0.8, rng=None):
    """
    Draws E from a lognormal. If rng is provided, use it; else fall back to np.random.
    """
    if rng is None:
        return float(np.random.lognormal(mean=mu, sigma=sigma))
    return float(rng.lognormal(mean=mu, sigma=sigma))

# Goldilocks modulation f(E,I)
def f_EI(E, I, E_c=E_C, sigma=SIGMA, alpha=ALPHA):
    """
    Gaussian window in linear E, scaled by (1 + alpha*I).
    Always non-negative; grows with I for fixed E proximity to E_c.
    """
    return np.exp(-(E - E_c)**2 / (2.0 * sigma**2)) * (1.0 + alpha * I)

# --- Per-block RNG setup: draw a local sub-seed and sync QuTiP/NumPy ---
sub_seed_collapse = int(master_rng.integers(0, 2**32 - 1))
rng_collapse = np.random.default_rng(sub_seed_collapse)
# QuTiP's rand_ket uses global NumPy RNG; keep it in sync for reproducibility
np.random.seed(sub_seed_collapse)

# Draw one (E, I) pair for the collapse demo (uses the local RNG + synced global)
E0 = sample_energy(rng=rng_collapse)
I0 = info_param()
f0 = f_EI(E0, I0)
X0 = E0 * I0 * f0

# Build the collapse trace around t=0:
#   - pre-collapse: larger fluctuations
#   - post-collapse: calmer dynamics
collapse_t = np.linspace(-0.2, 0.2, 200)
X_series = X0 + 0.5 * rng_collapse.normal(size=len(collapse_t))  # pre-collapse fluctuation
post_mask = (collapse_t >= 0)
X_series[post_mask] = X0 + 0.05 * rng_collapse.normal(size=post_mask.sum())  # post-collapse calm

# Plot and save
plt.figure()
plt.plot(collapse_t, X_series, "k-", alpha=0.6, label="fluctuation → lock-in")
plt.axhline(X0, color="r", ls="--", label=f"Lock-in X={X0:.2f}")
plt.axvline(0, color="r", lw=2)
plt.title("t = 0 : Collapse (E·I + Goldilocks)")
plt.xlabel("time (collapse)"); plt.ylabel("X = E·I·f(E,I)"); plt.legend()
savefig(os.path.join(FIG_DIR, "collapse.png"))

pd.DataFrame({"time": collapse_t, "X_vals": X_series}).to_csv(
    os.path.join(SAVE_DIR, "collapse.csv"), index=False
)

# ======================================================
# 3) Law lock-in model: c(t) stabilization with calmness check
#     – RNG-safe, MASTER_CTRL-driven thresholds, explicit E,I,f influence
# ======================================================

def law_lock_in(E, I, n_epoch=None, rng=None,
                f_min=0.1, target_X=5.0, base_noise=1e6, lock_eps_base=1e-3):
    """
    Simulate the stabilization ('lock-in') of a law proxy (e.g., speed of light c).
    - All randomness comes from `rng` (np.random.Generator). If None, a local one is used.
    - Thresholds are MASTER_CTRL-driven and softly modulated by f(E,I) and X=E*I*f.

    Returns
    -------
    (locked_at_epoch or -1, history_list)
    """
    # --- wire defaults from MASTER_CTRL ---
    if n_epoch is None:
        n_epoch = MASTER_CTRL.get("N_epoch", 500)
    if rng is None:
        rng = np.random.default_rng()

    # --- Goldilocks gate ---
    f = f_EI(E, I)
    if f < f_min:
        return -1, []

    # --- Coupled driver X and its squashed form for stable scaling ---
    X  = E * I * f
    Xn = X / (1.0 + X)  # in [0,1)

    # --- Initial c: narrower prior when f is larger (calmer environment) ---
    c_mean  = 3e8
    c_sigma = 1e7 * (1.1 - 0.3 * f)  # ~[0.8..1.1] scaling
    c_val   = float(rng.normal(c_mean, c_sigma))

    calm = 0
    locked_at = None
    history = []

    # --- Lock calmness threshold: loosen slightly at high f ---
    lock_eps = lock_eps_base * (1.1 - 0.5 * f)  # ≈ 1e-3 → [~5e-4..1.1e-3]
    lock_consec = MASTER_CTRL.get("lock_consecutive", 5)

    for n in range(n_epoch):
        prev = c_val

        # Noise shaped by proximity to target_X and damped by f and Xn
        #  - closer to target_X ⇒ smaller "shape"
        #  - higher f or Xn ⇒ more damping (calmer)
        shape = (1.0 + abs(X - target_X) / 10.0)
        damp  = (1.15 - 0.5 * f) * (1.05 - 0.4 * Xn)
        noise = base_noise * shape * damp * float(rng.uniform(0.8, 1.2))

        # Update c with adaptive noise
        c_val = c_val + float(rng.normal(0.0, noise))
        history.append(c_val)

        # Calmness check (relative step)
        delta = abs(c_val - prev) / max(abs(prev), 1e-9)
        if delta < lock_eps:
            calm += 1
            if calm >= lock_consec and locked_at is None:
                locked_at = n
        else:
            calm = 0

    return locked_at if locked_at is not None else -1, history


# ======================================================
# Stability predicate used in MC (amplitude calmness gated by f(E,I))
#     – RNG-safe, MASTER_CTRL thresholds, explicit E,I,f influence
# ======================================================
def is_stable(E, I, n_epoch=None, rel_eps=None, lock_consec=None, rng=None):
    """
    Returns 1 if the universe stabilizes; 0 otherwise.
    Dynamics depend on f(E,I) and X=E*I*f:
      - growth rate slightly increases with f and Xn
      - noise decreases as f grows (calmer near Goldilocks)
      - calmness threshold is MASTER_CTRL-driven and mildly modulated by f
    """
    # --- wire defaults from MASTER_CTRL ---
    if rng is None:
        rng = np.random.default_rng()
    if n_epoch is None:
        n_epoch = MASTER_CTRL.get("N_epoch", 500)
    if rel_eps is None:
        rel_eps = MASTER_CTRL.get("rel_eps", 0.05)
    if lock_consec is None:
        lock_consec = MASTER_CTRL.get("lock_consecutive", 5)

    # --- Goldilocks gate ---
    f = f_EI(E, I)
    if f < 0.2:
        return 0

    # --- Coupled driver X and squashed form ---
    X  = E * I * f
    Xn = X / (1.0 + X)

    # --- Initial amplitude and calm counter ---
    A, calm = 20.0, 0

    for _ in range(n_epoch):
        A_prev = A

        # Growth slightly boosted by f and Xn
        growth = 1.01 + 0.015 * f + 0.01 * Xn     # ~1.01 .. 1.035

        # Noise decreases with f (calmer near Goldilocks), with a floor
        noise_sigma = max(0.05, 2.0 * (1.2 - 0.6 * f))  # ~(2.4 .. 0.48), floored

        # Update amplitude
        A = A * growth + float(rng.normal(0.0, noise_sigma))

        # Effective calmness threshold (slightly looser for high f)
        delta   = abs(A - A_prev) / max(abs(A_prev), 1e-6)
        eps_eff = rel_eps * (1.1 - 0.4 * f)       # ≈ 5% → ~3–5%

        calm = calm + 1 if delta < eps_eff else 0
        if calm >= lock_consec:
            return 1

    return 0

# ======================================================
# 4) Monte Carlo: Stability + Law lock-in for many universes
#      – per-universe RNG, safe QuTiP seeding, fully reproducible
# ======================================================

def sample_I(dim=8):
    # Wrapper kept for readability; info_param() belül QuTiP-et hív.
    return info_param(dim=dim)

E_vals, I_vals, f_vals, X_vals = [], [], [], []
stables, law_epochs, final_cs, all_histories = [], [], [], []
sub_seeds = []  # store per-universe seeds for reproducibility

for _ in range(NUM_UNIVERSES):
    # --- draw a unique per-universe seed from the master RNG ---
    sub_seed = int(master_rng.integers(0, 2**32 - 1))
    sub_seeds.append(sub_seed)

    # --- per-universe modern Generator (use this everywhere we can) ---
    rng_uni = np.random.default_rng(sub_seed)

    # --- sample Energy with the per-universe RNG (no global RNG used here) ---
    Ei = float(rng_uni.lognormal(mean=2.5, sigma=0.8))

    # --- sample Information I with QuTiP (needs legacy RNG); do it safely ---
    # Save legacy RNG state, seed it to sub_seed for deterministic qt.rand_ket, then restore.
    legacy_state = np.random.get_state()
    np.random.seed(sub_seed)          # sync for qt.rand_ket() internals
    Ii = sample_I(dim=8)
    np.random.set_state(legacy_state) # restore previous global RNG state

    # --- compute Goldilocks modulation and coupled driver ---
    fi = f_EI(Ei, Ii)
    Xi = Ei * Ii * fi

    E_vals.append(Ei)
    I_vals.append(Ii)
    f_vals.append(fi)
    X_vals.append(Xi)

    # --- stability and lock-in, driven by the same per-universe RNG ---
    s = is_stable(Ei, Ii, rng=rng_uni)
    stables.append(s)

    if s == 1:
        lock_epoch, c_hist = law_lock_in(Ei, Ii, n_epoch=LOCKIN_EPOCHS, rng=rng_uni)
        law_epochs.append(lock_epoch)

        if c_hist:
            final_cs.append(c_hist[-1])
            all_histories.append(c_hist)
        else:
            final_cs.append(np.nan)
    else:
        law_epochs.append(-1)
        final_cs.append(np.nan)

# --- lock-in stats (guard empty) ---
valid_epochs = [e for e in law_epochs if e is not None and e >= 0]
median_epoch = float(np.median(valid_epochs)) if valid_epochs else None
mean_epoch   = float(np.mean(valid_epochs))   if valid_epochs else None

print(f"\n🔒 Universes with lock-in: {len(valid_epochs)} / {NUM_UNIVERSES}")

# --- Save per-universe seeds for reproducibility ---
pd.DataFrame(
    {"universe_id": np.arange(NUM_UNIVERSES), "seed": sub_seeds}
).to_csv(os.path.join(SAVE_DIR, "universe_seeds.csv"), index=False)

# ======================================================
# 5) Master DataFrame and saves
# ======================================================
df = pd.DataFrame({
    "E": E_vals,
    "I": I_vals,
    "fEI": f_vals,
    "X": X_vals,
    "stable": stables,
    "lock_epoch": law_epochs,
    "final_c": final_cs,
})
df.to_csv(os.path.join(SAVE_DIR, "tqe_runs.csv"), index=False)

# Diagnostics
stable_total = int(np.sum(stables))
valid_lockins = int(np.sum([e >= 0 for e in law_epochs]))
valid_lockins_among_stable = int(np.sum([e >= 0 for e, s in zip(law_epochs, stables) if s == 1]))

print("\n[DIAG] Stability vs Law lock-in")
print(f"Stable universes: {stable_total}/{NUM_UNIVERSES} ({100*stable_total/NUM_UNIVERSES:.1f}%)")
print(f"Lock-ins (any):   {valid_lockins}/{NUM_UNIVERSES} ({100*valid_lockins/NUM_UNIVERSES:.1f}%)")
if stable_total > 0:
    print(f"Lock-ins among stable: {valid_lockins_among_stable}/{stable_total} "
          f"({100*valid_lockins_among_stable/stable_total:.1f}%)")

# ======================================================
# 6) Stability summary — with lock-in split (E,I case)
# ======================================================
stable_total   = int(np.sum(stables))
unstable_total = int(NUM_UNIVERSES - stable_total)

# universes with lock-in (law_epochs >= 0)
locked_total = int(np.sum([e >= 0 for e in law_epochs]))
stable_no_lock = max(stable_total - locked_total, 0)

# percentages
pct_stable_no_lock = 100.0 * stable_no_lock / NUM_UNIVERSES
pct_locked         = 100.0 * locked_total / NUM_UNIVERSES
pct_unstable       = 100.0 * unstable_total / NUM_UNIVERSES

print("\n🌌 Universe Stability Summary (E,I)")
print(f"Total universes simulated: {NUM_UNIVERSES}")
print(f"Stable universes (lock-in):     {locked_total} ({pct_locked:.2f}%)")
print(f"Stable universes (no lock-in):  {stable_no_lock} ({pct_stable_no_lock:.2f}%)")
print(f"Unstable universes:             {unstable_total} ({pct_unstable:.2f}%)")

# --- plot stacked bar ---
plt.figure()

# x positions: 0 = Stable (stacked), 1 = Unstable
plt.bar(0, stable_no_lock, color="#1f77b4",
        label=f"Stable (no lock-in) [{stable_no_lock}, {pct_stable_no_lock:.1f}%]")
plt.bar(0, locked_total, bottom=stable_no_lock, color="#9467bd",
        label=f"Stable (lock-in) [{locked_total}, {pct_locked:.1f}%]")
plt.bar(1, unstable_total, color="#d62728",
        label=f"Unstable [{unstable_total}, {pct_unstable:.1f}%]")

plt.xticks([0, 1], ["Stable (split by lock-in)", "Unstable"])
plt.ylabel("Number of Universes")
plt.title("Universe Stability Distribution (E,I) — with Lock-in")
plt.legend(loc="upper right", frameon=False)
plt.tight_layout()
savefig(os.path.join(FIG_DIR, "stability_summary_with_lockin.png"))

# ======================================================
# 7) Average law lock-in dynamics across all universes
# ======================================================
if all_histories:
    min_len = min(len(h) for h in all_histories)
    truncated = [h[:min_len] for h in all_histories]
    avg_c = np.mean(truncated, axis=0)
    std_c = np.std(truncated, axis=0)

    pd.DataFrame({
        "epoch": np.arange(min_len),
        "avg_c": avg_c,
        "std_c": std_c
    }).to_csv(os.path.join(SAVE_DIR, "law_lockin_avg.csv"), index=False)

    if PLOT_AVG_LOCKIN and (median_epoch is not None):
        plt.figure()
        plt.plot(avg_c, label="Average c value")
        plt.fill_between(np.arange(min_len), avg_c-std_c, avg_c+std_c,
                         alpha=0.3, color="blue", label="±1σ")
        plt.axvline(median_epoch, color="r", ls="--", lw=2,
                    label=f"Median lock-in ≈ {median_epoch:.0f}")
        plt.title("Average law lock-in dynamics (Monte Carlo)")
        plt.xlabel("epoch"); plt.ylabel("c value (m/s)"); plt.legend()
        savefig(os.path.join(FIG_DIR, "law_lockin_avg.png"))

# ======================================================
# 8) t > 0 : Expansion dynamics (demo; uses single E0,I0 and median lock)
# ======================================================
def evolve(E, I, n_epoch=EXPANSION_EPOCHS):
    A_series, I_series = [], []
    A, orient = 20.0, I
    for _ in range(n_epoch):
        A = A * 1.005 + np.random.normal(0, 1.0)
        noise = 0.25 * (1 + 1.5 * abs(orient - 0.5))
        orient += (0.5 - orient) * 0.35 + np.random.normal(0, noise)
        orient = np.clip(orient, 0, 1)
        A_series.append(A); I_series.append(orient)
    return A_series, I_series

A_series, I_series = evolve(E0, I0, n_epoch=EXPANSION_EPOCHS)
plt.figure()
plt.plot(A_series, label="Amplitude A")
plt.plot(I_series, label="Orientation I")
plt.axhline(np.mean(A_series), color="gray", ls="--", alpha=0.5, label="Equilibrium A")
if median_epoch is not None:
    plt.axvline(median_epoch, color="r", ls="--", lw=2, label=f"Law lock-in ≈ {int(median_epoch)}")
    title_suffix = ""
else:
    title_suffix = " (no lock-in observed)"
plt.title("t > 0 : Expansion dynamics" + title_suffix)
plt.xlabel("epoch"); plt.ylabel("Parameters"); plt.legend()
savefig(os.path.join(FIG_DIR, "expansion.png"))

# Save expansion CSV
pd.DataFrame({
    "epoch": np.arange(len(A_series)),
    "Amplitude_A": A_series,
    "Orientation_I": I_series
}).to_csv(os.path.join(SAVE_DIR, "expansion.csv"), index=False)

# ======================================================
# 9) Histogram of lock-in epochs (CSV always, PNG optional)
# ======================================================
pd.DataFrame({"lock_epoch": valid_epochs}).to_csv(
    os.path.join(SAVE_DIR, "law_lockin_epochs.csv"), index=False
)

if PLOT_LOCKIN_HIST and len(valid_epochs) > 0:
    plt.figure()
    plt.hist(valid_epochs, bins=50, color="blue", alpha=0.7)
    if median_epoch is not None:
        plt.axvline(median_epoch, color="r", ls="--", lw=2,
                    label=f"Median lock-in = {int(median_epoch)}")
        plt.legend()
    plt.title("Distribution of law lock-in epochs (Monte Carlo)")
    plt.xlabel("Epoch of lock-in"); plt.ylabel("Count")
    savefig(os.path.join(FIG_DIR, "law_lockin_mc.png"))

# ======================================================
# 10) Save additional CSVs and summary.json
# ======================================================
# Stability outcomes CSV (compact)
pd.DataFrame({
    "E": E_vals,
    "I": I_vals,
    "X": X_vals,
    "Stable": stables,
    "lock_epoch": law_epochs,
    "final_c": final_cs
}).to_csv(os.path.join(SAVE_DIR, "stability.csv"), index=False)

summary = {
    "simulation": {
        "total_universes": NUM_UNIVERSES,
        "stable_fraction": float(np.mean(stables)),
        "unstable_fraction": 1.0 - float(np.mean(stables))
    },
    "seeds": {
    "master_seed": master_seed,
    "universe_seeds_csv": "universe_seeds.csv"
    },
    "superposition": {
        "mean_entropy": float(np.mean(S)),
        "mean_purity": float(np.mean(P))
    },
    "collapse": {
        "E0": float(E0),
        "I0": float(I0),
        "f0": float(f0),
        "mean_X": float(np.mean(X_vals)),
        "std_X": float(np.std(X_vals))
    },
    "law_lockin": {
        "mean_lock_epoch": float(np.mean(valid_epochs)) if len(valid_epochs) > 0 else None,
        "median_lock_epoch": float(np.median(valid_epochs)) if len(valid_epochs) > 0 else None,
        "locked_fraction": float(np.mean([1 if e >= 0 else 0 for e in law_epochs])),
        "mean_final_c": float(np.nanmean(final_cs)) if len(final_cs) > 0 else None,
        "std_final_c": float(np.nanstd(final_cs)) if len(final_cs) > 0 else None
    },
    "diag": {
        "stable_total": stable_total,
        "valid_lockins": valid_lockins,
        "valid_lockins_among_stable": valid_lockins_among_stable
    }
}
with open(os.path.join(SAVE_DIR, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

# ======================================================
# 10b) "Best universe" deep-dive — entropy plot like the screenshot
#       (pick earliest-locking stable universe, then simulate regions)
# ======================================================

# ----- Scoring rule -----
# among stable universes with lock-in (lock_epoch >= 0), pick the one with
# the *earliest* lock-in. if none locked, fall back to the highest f(E,I).
locked_idxs = [i for i, e in enumerate(law_epochs) if e >= 0 and stables[i] == 1]

if len(locked_idxs) > 0:
    # earliest lock-in epoch
    best_idx = locked_idxs[int(np.argmin([law_epochs[i] for i in locked_idxs]))]
    reason = f"earliest lock-in (epoch={law_epochs[best_idx]})"
else:
    # fallback: pick universe with the largest Goldilocks modulation f(E,I)
    best_idx = int(np.argmax(f_vals))
    reason = "no lock-ins → picked max f(E,I)"

E_best = E_vals[best_idx]
I_best = I_vals[best_idx]
print(f"[BEST] Universe index={best_idx} chosen by {reason}; E*={E_best:.3f}, I*={I_best:.3f}")

# ----- Single-universe entropy simulator (same style as your screenshot) -----

def simulate_entropy_universe(E, I, steps=BEST_STEPS,
                              num_regions=BEST_NUM_REGIONS, num_states=BEST_NUM_STATES):
    """
    Runs a single-universe entropy evolution with f(E,I) modulation.
    Returns: (region_entropies_list, global_entropy_list, lock_in_step)
    - region_entropies_list: list over time, each item is length `num_regions`
    - global_entropy_list: list over time (scalar entropy per step)
    - lock_in_step: first step where calmness criterion is satisfied (or None)
    """
    def f_EI_local(E, I, E_c=E_C, sigma=SIGMA, alpha=ALPHA):
        return np.exp(-(E - E_c)**2 / (2 * sigma**2)) * (1 + alpha * I)

    from scipy.stats import entropy

    # init states: break symmetry
    states = np.zeros((num_regions, num_states))
    states[0, :] = 1.0

    region_entropies, global_entropy = [], []
    lock_in_step, consecutive_calm = None, 0

    # dynamic vars
    A = 1.0
    orient = float(I)
    E_run = float(E)

    for step in range(steps):
        # cooling schedule for noise
        noise_scale = max(0.02, 1.0 - step / steps)

        # amplitude + orientation drift
        if step > 0:
            A = A * 1.01 + np.random.normal(0, 0.02)
            orient += (0.5 - orient) * 0.10 + np.random.normal(0, 0.02)
            orient = np.clip(orient, 0, 1)

        # energy random walk + recompute f
        E_run += np.random.normal(0, 0.05)
        f_step_base = f_EI_local(E_run, I)

        # update regions with noise and occasional shocks
        for r in range(num_regions):
            noise = np.random.normal(0, noise_scale * 5.0, num_states)
            if np.random.rand() < 0.05:
                noise += np.random.normal(0, 8.0, num_states)
            f_step = f_step_base * (1 + np.random.normal(0, 0.1))
            states[r] += f_step * noise
            states[r] = np.clip(states[r], 0, 1)

        # entropies
        region_entropies.append([entropy(states[r]) for r in range(num_regions)])
        global_entropy.append(entropy(states.flatten()))

        # lock-in detection: calm relative change of global entropy
        if step > 0:
            prev = global_entropy[-2]
            cur  = global_entropy[-1]
            delta = abs(cur - prev) / max(prev, 1e-9)
            if delta < 0.001:
                consecutive_calm += 1
                if consecutive_calm >= 10 and lock_in_step is None:
                    lock_in_step = step
            else:
                consecutive_calm = 0

    return region_entropies, global_entropy, lock_in_step

# ----- Run the deep-dive sim on the chosen (E*, I*) -----
best_region_entropies, best_global_entropy, best_lock = simulate_entropy_universe(E_best, I_best)

# ----- Save CSVs -----
# global entropy
pd.DataFrame({
    "time": np.arange(len(best_global_entropy)),
    "global_entropy": best_global_entropy
}).to_csv(os.path.join(SAVE_DIR, "best_universe_global_entropy.csv"), index=False)

# per-region entropy (wide)
best_re_mat = np.array(best_region_entropies)  # shape: (steps, regions)
re_cols = [f"region_{i}_entropy" for i in range(best_re_mat.shape[1])]
pd.DataFrame(best_re_mat, columns=re_cols).assign(time=np.arange(best_re_mat.shape[0])) \
  .to_csv(os.path.join(SAVE_DIR, "best_universe_region_entropies.csv"), index=False)

# ----- Plot like the screenshot -----
plt.figure(figsize=(12, 6))
time_axis = np.arange(len(best_global_entropy))

# region curves
for r in range(min(BEST_NUM_REGIONS, best_re_mat.shape[1])):
    plt.plot(time_axis, best_re_mat[:, r], lw=1, label=f"Region {r} entropy")

# global + thresholds
plt.plot(time_axis, best_global_entropy, color="black", linewidth=2, label="Global entropy")
plt.axhline(y=STABILITY_THRESHOLD, color="red", linestyle="--", label="Stability threshold")

# lock-in indicator
if best_lock is not None:
    plt.axvline(x=best_lock, color="purple", linestyle="--", linewidth=2,
                label=f"Lock-in step = {best_lock}")

plt.title("Best-universe entropy evolution (chosen from MC)")
plt.xlabel("Time step"); plt.ylabel("Entropy"); plt.legend(ncol=2)
plt.grid(True, alpha=0.3)
savefig(os.path.join(FIG_DIR, "best_universe_entropy_evolution.png"))

# ======================================================
# 11) XAI (SHAP + LIME) — classification on stability, regression on lock_epoch
# ======================================================
if RUN_XAI:
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.metrics import r2_score, accuracy_score

    # ---------- Features & targets ----------
    X_feat = df[["E", "I", "X"]].copy()
    y_cls  = df["stable"].astype(int).values

    reg_mask = df["lock_epoch"] >= 0
    X_reg = X_feat[reg_mask]
    y_reg = df.loc[reg_mask, "lock_epoch"].values

    # ---------- Guard: check if both classes exist ---------- 
    uniq_vals, uniq_cnts = np.unique(y_cls, return_counts=True)
    have_two_classes = (len(uniq_vals) == 2)

    # ---------- Train/Test split (classification) ----------
    can_stratify = have_two_classes and (uniq_cnts.min() >= 2)
    stratify_arg = y_cls if can_stratify else None
    if not can_stratify:
        print(f"[XAI] Stratify disabled (class counts = {dict(zip(uniq_vals, uniq_cnts))})")

    # If only a single class exists in the entire sample, skip classification
    do_classification = have_two_classes

    if do_classification:
        Xtr_c, Xte_c, ytr_c, yte_c = train_test_split(
            X_feat, y_cls, test_size=0.25, random_state=42, stratify=stratify_arg
        )

    # ---------- Regression split (lock_epoch) ----------
    have_reg = len(X_reg) >= 30
    if have_reg:
        Xtr_r, Xte_r, ytr_r, yte_r = train_test_split(
            X_reg, y_reg, test_size=0.25, random_state=42
        )

    # ---------- Train models ----------
    if do_classification:
        rf_cls = RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1)
        rf_cls.fit(Xtr_c, ytr_c)
        cls_acc = accuracy_score(yte_c, rf_cls.predict(Xte_c))
        print(f"[XAI] Classification accuracy (stable): {cls_acc:.3f}")
    else:
        rf_cls = None
        print("[XAI] Skipping classification (only one class present).")

    if have_reg:
        rf_reg = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)
        rf_reg.fit(Xtr_r, ytr_r)
        reg_r2 = r2_score(yte_r, rf_reg.predict(Xte_r))
        print(f"[XAI] Regression R^2 (lock_epoch): {reg_r2:.3f}")
    else:
        rf_reg, reg_r2 = None, None
        print("[XAI] Not enough locked samples for regression (need ~30+).")

    # ---------- SHAP plots & CSV ----------
    if do_classification:
        X_plot = Xte_c.copy()
        try:
            expl_cls = shap.TreeExplainer(
                rf_cls, feature_perturbation="interventional", model_output="raw"
            )
            sv_cls = expl_cls.shap_values(X_plot, check_additivity=False)
        except Exception:
            expl_cls = shap.Explainer(rf_cls, Xtr_c)
            sv_cls = expl_cls(X_plot).values

        if isinstance(sv_cls, list):
            sv_cls = sv_cls[1]  # positive class
        sv_cls = np.asarray(sv_cls)
        if sv_cls.ndim == 3 and sv_cls.shape[0] == X_plot.shape[0]:
            sv_cls = sv_cls[:, :, 1]
        elif sv_cls.ndim == 3 and sv_cls.shape[-1] == X_plot.shape[1]:
            sv_cls = sv_cls[1, :, :]
        assert sv_cls.shape == X_plot.shape, f"SHAP shape {sv_cls.shape} != data {X_plot.shape}"

        plt.figure()
        shap.summary_plot(sv_cls, X_plot.values, feature_names=X_plot.columns.tolist(), show=False)
        plt.title("SHAP summary – classification (stable)")
        plt.savefig(os.path.join(FIG_DIR, "shap_summary_cls_stable.png"), dpi=220, bbox_inches="tight")
        plt.close()

        pd.DataFrame(np.asarray(sv_cls), columns=X_plot.columns).to_csv(
            os.path.join(FIG_DIR, "shap_values_classification.csv"), index=False
        )
        cls_importance = pd.Series(np.mean(np.abs(sv_cls), axis=0), index=X_plot.columns)\
                         .sort_values(ascending=False)
        cls_importance.to_csv(
            os.path.join(FIG_DIR, "shap_feature_importance_classification.csv"),
            header=["mean_|shap|"]
        )

    if rf_reg is not None:
        X_plot_r = Xte_r.copy()
        try:
            expl_reg = shap.TreeExplainer(
                rf_reg, feature_perturbation="interventional", model_output="raw"
            )
            sv_reg = expl_reg.shap_values(X_plot_r, check_additivity=False)
        except Exception:
            expl_reg = shap.Explainer(rf_reg, Xtr_r)
            sv_reg = expl_reg(X_plot_r).values

        sv_reg = np.asarray(sv_reg)
        if sv_reg.ndim == 3 and sv_reg.shape[0] == X_plot_r.shape[0]:
            sv_reg = sv_reg[:, :, 0]
        elif sv_reg.ndim == 3 and sv_reg.shape[-1] == X_plot_r.shape[1]:
            sv_reg = sv_reg[0, :, :]
        assert sv_reg.shape == X_plot_r.shape, f"SHAP shape {sv_reg.shape} != data {X_plot_r.shape}"

        plt.figure()
        shap.summary_plot(sv_reg, X_plot_r.values, feature_names=X_plot_r.columns.tolist(), show=False)
        plt.title("SHAP summary – regression (lock_epoch)")
        plt.savefig(os.path.join(FIG_DIR, "shap_summary_reg_lock_at.png"), dpi=220, bbox_inches="tight")
        plt.close()

        pd.DataFrame(sv_reg, columns=X_plot_r.columns).to_csv(
            os.path.join(FIG_DIR, "shap_values_regression_lock_at.csv"), index=False
        )
        reg_importance = pd.Series(np.mean(np.abs(sv_reg), axis=0), index=X_plot_r.columns)\
                         .sort_values(ascending=False)
        reg_importance.to_csv(
            os.path.join(FIG_DIR, "shap_feature_importance_regression_lock_at.csv"),
            header=["mean_|shap|"]
        )

    # ---------- LIME (only if classification ran) ----------
    if do_classification:
        lime_explainer = LimeTabularExplainer(
            training_data=X_feat.values,
            feature_names=X_feat.columns.tolist(),
            discretize_continuous=True,
            mode='classification'
        )
        exp = lime_explainer.explain_instance(Xte_c.iloc[0].values, rf_cls.predict_proba, num_features=5)
        lime_list = exp.as_list(label=1)
        pd.DataFrame(lime_list, columns=["feature", "weight"]).to_csv(
            os.path.join(FIG_DIR, "lime_example_classification.csv"), index=False
        )

print("\n✅ DONE.")
print(f"☁️ All results saved to Google Drive: {SAVE_DIR}")
