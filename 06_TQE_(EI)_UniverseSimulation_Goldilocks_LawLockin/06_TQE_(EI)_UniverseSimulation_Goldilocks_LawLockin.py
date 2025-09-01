# ===========================================================================
# Theory of the Question of Existence (TQE)
# (E, I) Vacuum fluctuation → Superposition → Collapse → Expansion → Law lock-in
# ===========================================================================
# Author: Stefan Len
# Description: Full Monte Carlo pipeline with Energy–Information (E,I) dynamics.
# Includes: QuTiP quantum stage, MC universes, stability & law lock-in,
#           averaged c(t), expansion, best-universe entropy deep-dive,
#           CSV/PNG artifacts, summary.json, SHAP/LIME diagnostics.
# ===========================================================================

# ---- Mount Google Drive (Colab) ----
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# ---- Minimal auto-install (safe, quiet) ----
import sys, subprocess, warnings
warnings.filterwarnings("ignore")

def _ensure(pkg):
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

# Needed packages (SciPy required by the best-universe entropy deep-dive)
for pkg in ["qutip", "pandas", "scikit-learn", "shap", "lime", "scipy", "matplotlib", "numpy"]:
    _ensure(pkg)

# ---- Imports ----
import os, time, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import qutip as qt
import shap
from lime.lime_tabular import LimeTabularExplainer

# ---- Directories ----
GOOGLE_BASE = "/content/drive/MyDrive/TQE_(E,I)_UNIVERSE_SIMULATION"
run_id = time.strftime("TQE_(E,I)_UNIVERSE_SIMULATION_%Y%m%d_%H%M%S")
SAVE_DIR = os.path.join(GOOGLE_BASE, run_id); os.makedirs(SAVE_DIR, exist_ok=True)
FIG_DIR  = os.path.join(SAVE_DIR, "figs");    os.makedirs(FIG_DIR,  exist_ok=True)

def savefig(path):
    """Safe figure save helper."""
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()

# ======================================================
# MASTER CONTROLLER – everything in one place
# ======================================================
MASTER_CTRL = {
    # ---- Core sizes ----
    "NUM_UNIVERSES":        5000,
    "TIME_STEPS":           1000,   # used by stability() loop length
    "LOCKIN_EPOCHS":        500,    # used by law_lock_in()
    "EXPANSION_EPOCHS":     1000,   # used by evolve()

    # ---- Energy distribution (log-space) ----
    "E_LOG_MU":             2.5,
    "E_LOG_SIGMA":          0.8,

    # ---- Goldilocks window & info coupling (linear E) ----
    "E_CENTER":             2.0,
    "E_WIDTH":              0.5,
    "ALPHA_I":              0.8,

    # ---- Stability / lock-in thresholds ----
    "F_GATE_STABLE":        0.20,
    "F_GATE_LOCKIN":        0.10,
    "CALM_STEPS_STABLE":    5,
    "CALM_STEPS_LOCKIN":    5,
    "REL_EPS_STABLE":       0.05,   # relative step threshold for stability()
    "REL_EPS_LOCKIN":       1e-3,   # relative step threshold for law_lock_in()

    # ---- Expansion dynamics ----
    "EXP_GROWTH_BASE":      1.005,
    "EXP_NOISE_BASE":       1.0,

    # ---- Quantum (superposition) block ----
    "Q_NLEV":               12,     # Hilbert-space size
    "Q_TMAX":               10.0,   # time max for plotting S/P
    "Q_NT":                 200,    # number of time samples
    "Q_GAMMA_BASE":         0.02,
    "Q_GAMMA_SIN":          0.01,
    "Q_GAMMA_NOISE":        0.005,
    "Q_SMALL_NOISE":        0.05,   # H0 perturbation scale
    "Q_WINDOW":             0.5,    # short mesolve window
    "Q_WINDOW_STEPS":       5,

    # ---- Collapse block ----
    "COLLAPSE_TMIN":        -0.2,
    "COLLAPSE_TMAX":        0.2,
    "COLLAPSE_N":           200,
    "COLLAPSE_NOISE_PRE":   0.5,
    "COLLAPSE_NOISE_POST":  0.05,

    # ---- Law lock-in shaping ----
    "LL_TARGET_X":          5.0,
    "LL_BASE_NOISE":        1e6,

    # ---- Best-universe deep dive ----
    "BEST_STEPS":           1000,
    "BEST_NUM_REGIONS":     10,
    "BEST_NUM_STATES":      500,
    "ENTROPY_STAB_THRESH":  3.5,
    "ENTROPY_CALM_EPS":     1e-3,
    "ENTROPY_CALM_CONSEC":  10,

    # ---- Feature flags ----
    "RUN_XAI":              True,
    "PLOT_AVG_LOCKIN":      False,
    "PLOT_LOCKIN_HIST":     False,
}

# ----- Aliases (read once) -----
NUM_UNIVERSES        = MASTER_CTRL["NUM_UNIVERSES"]
LOCKIN_EPOCHS        = MASTER_CTRL["LOCKIN_EPOCHS"]
EXPANSION_EPOCHS     = MASTER_CTRL["EXPANSION_EPOCHS"]

E_LOG_MU             = MASTER_CTRL["E_LOG_MU"]
E_LOG_SIGMA          = MASTER_CTRL["E_LOG_SIGMA"]
E_CENTER             = MASTER_CTRL["E_CENTER"]
E_WIDTH              = MASTER_CTRL["E_WIDTH"]
ALPHA_I              = MASTER_CTRL["ALPHA_I"]

F_GATE_STABLE        = MASTER_CTRL["F_GATE_STABLE"]
F_GATE_LOCKIN        = MASTER_CTRL["F_GATE_LOCKIN"]
REL_EPS_STABLE       = MASTER_CTRL["REL_EPS_STABLE"]
REL_EPS_LOCKIN       = MASTER_CTRL["REL_EPS_LOCKIN"]
CALM_STEPS_STABLE    = MASTER_CTRL["CALM_STEPS_STABLE"]
CALM_STEPS_LOCKIN    = MASTER_CTRL["CALM_STEPS_LOCKIN"]

RUN_XAI              = MASTER_CTRL["RUN_XAI"]
PLOT_AVG_LOCKIN      = MASTER_CTRL["PLOT_AVG_LOCKIN"]
PLOT_LOCKIN_HIST     = MASTER_CTRL["PLOT_LOCKIN_HIST"]

# --- Master RNG (reproducibility) ---
master_seed = int(np.random.SeedSequence().generate_state(1)[0])
master_rng  = np.random.default_rng(master_seed)
print(f"[SEED] master_seed = {master_seed}")

# ======================================================
# Shared helpers (E,I,f)
# ======================================================

def KL(p, q, eps=1e-12):
    """KL divergence with clipping and renormalization (safe)."""
    p = np.clip(p, eps, None); q = np.clip(q, eps, None)
    p = p / p.sum(); q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))

def info_param(dim=8):
    """
    Information parameter I in [0,1], built from KL × normalized Shannon entropy.
    Uses QuTiP's rand_ket (depends on NumPy global RNG); seed np.random before call if needed.
    """
    psi1, psi2 = qt.rand_ket(dim), qt.rand_ket(dim)
    p1 = np.abs(psi1.full().ravel())**2
    p2 = np.abs(psi2.full().ravel())**2

    KL_val = KL(p1, p2)
    I_kl   = KL_val / (1.0 + KL_val)

    eps = 1e-12
    H    = -np.sum(p1 * np.log(p1 + eps))
    I_sh = H / np.log(len(p1))

    I_raw = I_kl * I_sh
    return I_raw / (1.0 + I_raw)

def sample_energy(rng=None, mu=E_LOG_MU, sigma=E_LOG_SIGMA):
    """Draw E from a lognormal (log-space mu, sigma)."""
    if rng is None:
        return float(np.random.lognormal(mean=mu, sigma=sigma))
    return float(rng.lognormal(mean=mu, sigma=sigma))

def f_EI(E, I, E_c=E_CENTER, sigma=E_WIDTH, alpha=ALPHA_I):
    """Gaussian Goldilocks in linear E, scaled by (1 + alpha*I)."""
    return np.exp(-((E - E_c) ** 2) / (2.0 * sigma ** 2)) * (1.0 + alpha * I)

# ======================================================
# 1) t < 0 : Quantum superposition (vacuum fluctuation)
# ======================================================

# --- Per-block RNG & QuTiP sync ---
sub_seed_super = int(master_rng.integers(0, 2**32 - 1))
rng_super = np.random.default_rng(sub_seed_super)
np.random.seed(sub_seed_super)

Nlev = MASTER_CTRL["Q_NLEV"]
a = qt.destroy(Nlev)

# Perturbed Hamiltonian with small random noise
H0 = a.dag()*a + MASTER_CTRL["Q_SMALL_NOISE"] * (rng_super.normal() * a + rng_super.normal() * a.dag())

# Initial state
psi0 = qt.rand_ket(Nlev)
rho0 = psi0 * psi0.dag()

# Time grid & gamma(t)
tlist  = np.linspace(0, MASTER_CTRL["Q_TMAX"], MASTER_CTRL["Q_NT"])
gammas = (MASTER_CTRL["Q_GAMMA_BASE"]
          + MASTER_CTRL["Q_GAMMA_SIN"] * np.sin(0.5 * tlist)
          + MASTER_CTRL["Q_GAMMA_NOISE"] * rng_super.normal(size=len(tlist)))

states = []
for g in gammas:
    res = qt.mesolve(H0, rho0,
                     np.linspace(0, MASTER_CTRL["Q_WINDOW"], MASTER_CTRL["Q_WINDOW_STEPS"]),
                     [np.sqrt(abs(g))*a], [], progress_bar=None)
    states.append(res.states[-1])

def purity(r):
    """Purity for density matrices and kets."""
    return float((r*r).tr().real) if qt.isoper(r) else float((r*r.dag()).tr().real)

S = np.array([qt.entropy_vn(r) for r in states], dtype=float)
P = np.array([purity(r)       for r in states], dtype=float)

plt.figure()
plt.plot(tlist, S, label="Entropy")
plt.plot(tlist, P, label="Purity")
plt.title("t < 0 : Quantum superposition (vacuum fluctuation)")
plt.xlabel("time"); plt.legend()
savefig(os.path.join(FIG_DIR, "superposition.png"))

pd.DataFrame({"time": tlist, "Entropy": S, "Purity": P}).to_csv(
    os.path.join(SAVE_DIR, "superposition.csv"), index=False
)

# ======================================================
# 2) t = 0 : Collapse (E·I coupling + Goldilocks factor)
# ======================================================

# Local sub-seed for collapse demo (sync QuTiP)
sub_seed_collapse = int(master_rng.integers(0, 2**32 - 1))
rng_collapse = np.random.default_rng(sub_seed_collapse)
np.random.seed(sub_seed_collapse)

# Draw one (E, I) and compute coupled driver
E0 = sample_energy(rng=rng_collapse)
I0 = info_param()
f0 = f_EI(E0, I0)
X0 = E0 * I0 * f0

tmin, tmax = MASTER_CTRL["COLLAPSE_TMIN"], MASTER_CTRL["COLLAPSE_TMAX"]
Npts       = MASTER_CTRL["COLLAPSE_N"]
collapse_t = np.linspace(tmin, tmax, Npts)

# Build collapse trace (pre: noisy, post: calm)
X_series = X0 + MASTER_CTRL["COLLAPSE_NOISE_PRE"] * rng_collapse.normal(size=Npts)
post_mask = (collapse_t >= 0)
X_series[post_mask] = X0 + MASTER_CTRL["COLLAPSE_NOISE_POST"] * rng_collapse.normal(size=post_mask.sum())

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
# 3) Law lock-in model & Stability predicate (MASTER_CTRL-driven)
# ======================================================

def law_lock_in(E, I, n_epoch=None, rng=None,
                f_min=F_GATE_LOCKIN, target_X=None, base_noise=None, lock_eps_base=REL_EPS_LOCKIN):
    """Simulate law lock-in for a proxy c(t); return (locked_epoch or -1, history list)."""
    if n_epoch is None:
        n_epoch = LOCKIN_EPOCHS
    if rng is None:
        rng = np.random.default_rng()

    if target_X is None:
        target_X = MASTER_CTRL["LL_TARGET_X"]
    if base_noise is None:
        base_noise = MASTER_CTRL["LL_BASE_NOISE"]

    f = f_EI(E, I)
    if f < f_min:
        return -1, []

    X  = E * I * f
    Xn = X / (1.0 + X)

    c_mean, c_sigma = 3e8, 1e7 * (1.1 - 0.3 * f)
    c_val = float(rng.normal(c_mean, c_sigma))

    history = [c_val]  # <-- start collecting history
    calm = 0
    locked_at = None
    lock_eps   = lock_eps_base * (1.1 - 0.5 * f)
    lock_consec = CALM_STEPS_LOCKIN

    for n in range(n_epoch):
        prev  = c_val
        shape = (1.0 + abs(X - target_X) / 10.0)
        damp  = (1.15 - 0.5 * f) * (1.05 - 0.4 * Xn)
        noise = base_noise * shape * damp * float(rng.uniform(0.8, 1.2))
        c_val = c_val + float(rng.normal(0.0, noise))

        history.append(c_val)  # <-- keep it

        delta = abs(c_val - prev) / max(abs(prev), 1e-9)
        if delta < lock_eps:
            calm += 1
            if calm >= lock_consec and locked_at is None:
                locked_at = n
        else:
            calm = 0

    return locked_at if locked_at is not None else -1, history

def is_stable(E, I, n_epoch=None, rel_eps=None, lock_consec=None, rng=None):
    """Return 1 if amplitude dynamics stabilize; else 0."""
    if rng is None:
        rng = np.random.default_rng()
    if n_epoch is None:
        n_epoch = MASTER_CTRL["TIME_STEPS"]
    if rel_eps is None:
        rel_eps = REL_EPS_STABLE
    if lock_consec is None:
        lock_consec = CALM_STEPS_STABLE

    f = f_EI(E, I)
    if f < F_GATE_STABLE:
        return 0

    X  = E * I * f
    Xn = X / (1.0 + X)

    A, calm = 20.0, 0
    for _ in range(n_epoch):
        A_prev = A
        growth      = 1.01 + 0.015 * f + 0.01 * Xn
        noise_sigma = max(0.05, 2.0 * (1.2 - 0.6 * f))
        A = A * growth + float(rng.normal(0.0, noise_sigma))

        delta   = abs(A - A_prev) / max(abs(A_prev), 1e-6)
        eps_eff = rel_eps * (1.1 - 0.4 * f)

        calm = calm + 1 if delta < eps_eff else 0
        if calm >= lock_consec:
            return 1

    return 0

# ======================================================
# 4) Monte Carlo: Stability + Law lock-in for many universes
# ======================================================
def sample_I(dim=8):
    """Wrapper for readability; relies on QuTiP + global RNG."""
    return info_param(dim=dim)

E_vals, I_vals, f_vals, X_vals = [], [], [], []
stables, law_epochs, final_cs, all_histories = [], [], [], []
sub_seeds = []

for _ in range(NUM_UNIVERSES):
    sub_seed = int(master_rng.integers(0, 2**32 - 1))
    sub_seeds.append(sub_seed)

    rng_uni = np.random.default_rng(sub_seed)

    # Energy via per-universe Generator
    Ei = sample_energy(rng=rng_uni)

    # I via QuTiP (temporarily seed global RNG)
    legacy_state = np.random.get_state()
    np.random.seed(sub_seed)
    Ii = sample_I(dim=8)
    np.random.set_state(legacy_state)

    fi = f_EI(Ei, Ii)
    Xi = Ei * Ii * fi

    E_vals.append(Ei); I_vals.append(Ii); f_vals.append(fi); X_vals.append(Xi)

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

valid_epochs = [e for e in law_epochs if e is not None and e >= 0]
median_epoch = float(np.median(valid_epochs)) if valid_epochs else None
mean_epoch   = float(np.mean(valid_epochs))   if valid_epochs else None

print(f"\n🔒 Universes with lock-in: {len(valid_epochs)} / {NUM_UNIVERSES}")

pd.DataFrame({"universe_id": np.arange(NUM_UNIVERSES), "seed": sub_seeds}).to_csv(
    os.path.join(SAVE_DIR, "universe_seeds.csv"), index=False
)

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
# --- Three-bar chart: Lock-in, Stable, Unstable (no legend; labels under bars) ---
total = int(len(df))
stable_count   = int(np.sum(np.asarray(stables, dtype=int)))           # all stable (with or without lock-in)
unstable_count = max(0, total - stable_count)                          # all unstable
lockin_count   = int(np.sum(np.asarray(law_epochs) >= 0))              # universes with law lock-in

# --- Safe denominator to avoid division by zero ---
den = max(1, total)

# Percentages (one decimal), also keep float values for CSV
p_lockin   = 100.0 * lockin_count   / den
p_stable   = 100.0 * stable_count   / den
p_unstable = 100.0 * unstable_count / den

# Labels shown *under* the bars
xtick_labels = [
    f"Lock-in ({lockin_count}, {p_lockin:.1f}%)",
    f"Stable ({stable_count}, {p_stable:.1f}%)",
    f"Unstable ({unstable_count}, {p_unstable:.1f}%)",
]

# Values in the same order
yvals = [lockin_count, stable_count, unstable_count]

# --- Plot ---
plt.figure()
plt.bar([0, 1, 2], yvals, color=["#6baed6", "#2ca02c", "#d62728"])
plt.xticks([0, 1, 2], xtick_labels, rotation=0)
plt.ylabel("Number of Universes")
plt.title("Universe Stability Distribution (E,I) — three categories")

# Optional: light grid for readability
plt.grid(axis="y", alpha=0.2)
plt.ylim(bottom=0)  # always start from zero

savefig(os.path.join(FIG_DIR, "stability_three_bars.png"))

# --- Save counts used in the 3-bar chart ---
pd.DataFrame(
    {
        "metric":  ["lock_in", "stable", "unstable"],
        "count":   [lockin_count, stable_count, unstable_count],
        "percent": [p_lockin,     p_stable,     p_unstable],
        "total":   [total,        total,        total],
    }
).to_csv(os.path.join(SAVE_DIR, "stability_three_bars.csv"), index=False)

# ======================================================
# 7) Average law lock-in dynamics across all universes (optional plot)
# ======================================================
if all_histories:
    min_len   = min(len(h) for h in all_histories)
    truncated = [h[:min_len] for h in all_histories]
    avg_c     = np.mean(truncated, axis=0)
    std_c     = np.std(truncated, axis=0)

    pd.DataFrame({"epoch": np.arange(min_len), "avg_c": avg_c, "std_c": std_c}).to_csv(
        os.path.join(SAVE_DIR, "law_lockin_avg.csv"), index=False
    )

    if PLOT_AVG_LOCKIN and (median_epoch is not None):
        plt.figure()
        plt.plot(avg_c, label="Average c value")
        plt.fill_between(np.arange(min_len), avg_c-std_c, avg_c+std_c, alpha=0.3, color="blue", label="±1σ")
        plt.axvline(median_epoch, color="r", ls="--", lw=2, label=f"Median lock-in ≈ {median_epoch:.0f}")
        plt.title("Average law lock-in dynamics (Monte Carlo)")
        plt.xlabel("epoch"); plt.ylabel("c value (m/s)"); plt.legend()
        savefig(os.path.join(FIG_DIR, "law_lockin_avg.png"))

# ======================================================
# 8) t > 0 : Expansion dynamics (demo; uses E0,I0)
# ======================================================
def evolve(E, I, n_epoch=None, rng=None):
    """Simulate post-lock expansion dynamics."""
    if n_epoch is None:
        n_epoch = EXPANSION_EPOCHS
    if rng is None:
        rng = np.random.default_rng()

    A_series, I_series = [], []
    A, orient = 20.0, I
    for _ in range(n_epoch):
        A = A * MASTER_CTRL["EXP_GROWTH_BASE"] + float(rng.normal(0.0, MASTER_CTRL["EXP_NOISE_BASE"]))
        noise = 0.25 * (1 + 1.5 * abs(orient - 0.5))
        orient += (0.5 - orient) * 0.35 + float(rng.normal(0.0, noise))
        orient = float(np.clip(orient, 0.0, 1.0))
        A_series.append(A); I_series.append(orient)
    return A_series, I_series

A_series, I_series = evolve(E0, I0, n_epoch=EXPANSION_EPOCHS, rng=master_rng)
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
    plt.hist(valid_epochs, bins=min(50, len(valid_epochs)), color="blue", alpha=0.7)
    if median_epoch is not None:
        plt.axvline(median_epoch, color="r", ls="--", lw=2, label=f"Median lock-in = {int(median_epoch)}")
        plt.legend()
    plt.title("Distribution of law lock-in epochs (Monte Carlo)")
    plt.xlabel("Epoch of lock-in"); plt.ylabel("Count")
    savefig(os.path.join(FIG_DIR, "law_lockin_mc.png"))

# ======================================================
# 10) Best-universe deep-dive (entropy plot)
# ======================================================
locked_idxs = [i for i, e in enumerate(law_epochs) if e >= 0 and stables[i] == 1]
if len(locked_idxs) > 0:
    best_idx = locked_idxs[int(np.argmin([law_epochs[i] for i in locked_idxs]))]
    reason = f"earliest lock-in (epoch={law_epochs[best_idx]})"
else:
    best_idx = int(np.argmax(f_vals))
    reason = "no lock-ins → picked max f(E,I)"

E_best, I_best = E_vals[best_idx], I_vals[best_idx]
print(f"[BEST] Universe index={best_idx} chosen by {reason}; E*={E_best:.3f}, I*={I_best:.3f}")

def simulate_entropy_universe(E, I,
                              steps=MASTER_CTRL["BEST_STEPS"],
                              num_regions=MASTER_CTRL["BEST_NUM_REGIONS"],
                              num_states=MASTER_CTRL["BEST_NUM_STATES"]):
    """Entropy evolution with f(E,I) modulation; returns (regions, global, lock_step)."""
    states = np.zeros((num_regions, num_states)); states[0, :] = 1.0
    region_entropies, global_entropy = [], []
    lock_in_step, consecutive_calm = None, 0
    A, orient, E_run = 1.0, float(I), float(E)

    for step in range(steps):
        noise_scale = max(0.02, 1.0 - step / steps)
        if step > 0:
            A = A * 1.01 + np.random.normal(0, 0.02)
            orient += (0.5 - orient) * 0.10 + np.random.normal(0, 0.02)
            orient = np.clip(orient, 0, 1)

        E_run += np.random.normal(0, 0.05)
        f_step_base = f_EI(E_run, I)

        for r in range(num_regions):
            noise = np.random.normal(0, noise_scale * 5.0, num_states)
            if np.random.rand() < 0.05:
                noise += np.random.normal(0, 8.0, num_states)
            f_step = f_step_base * (1 + np.random.normal(0, 0.1))
            states[r] = np.clip(states[r] + f_step * noise, 0, 1)

        region_entropies.append([_entropy(states[r]) for r in range(num_regions)])
        global_entropy.append(_entropy(states.flatten()))

        if step > 0:
            prev, cur = global_entropy[-2], global_entropy[-1]
            delta = abs(cur - prev) / max(prev, 1e-9)
            if delta < MASTER_CTRL["ENTROPY_CALM_EPS"]:
                consecutive_calm += 1
                if consecutive_calm >= MASTER_CTRL["ENTROPY_CALM_CONSEC"] and lock_in_step is None:
                    lock_in_step = step
            else:
                consecutive_calm = 0

    return region_entropies, global_entropy, lock_in_step

best_region_entropies, best_global_entropy, best_lock = simulate_entropy_universe(E_best, I_best)

# Save CSVs
best_re_df = pd.DataFrame(best_re_mat, columns=re_cols)
best_re_df.insert(0, "time", np.arange(best_re_mat.shape[0]))
best_re_df.to_csv(os.path.join(SAVE_DIR, "best_universe_region_entropies.csv"), index=False)
)
best_re_mat = np.array(best_region_entropies)
re_cols = [f"region_{i}_entropy" for i in range(best_re_mat.shape[1])]
pd.DataFrame(best_re_mat, columns=re_cols).assign(time=np.arange(best_re_mat.shape[0])).to_csv(
    os.path.join(SAVE_DIR, "best_universe_region_entropies.csv"), index=False
)

# Plot
plt.figure(figsize=(12, 6))
time_axis = np.arange(len(best_global_entropy))
for r in range(min(10, best_re_mat.shape[1])):
    plt.plot(time_axis, best_re_mat[:, r], lw=1, label=f"Region {r} entropy")
plt.plot(time_axis, best_global_entropy, color="black", linewidth=2, label="Global entropy")
thr = MASTER_CTRL["ENTROPY_STAB_THRESH"]
plt.axhline(y=thr, color="red", linestyle="--", label="Stability threshold")
# annotate calmness rule used for lock detection
plt.text(0.01, 0.02,
         f"calmness: eps={MASTER_CTRL['ENTROPY_CALM_EPS']}, "
         f"consec={MASTER_CTRL['ENTROPY_CALM_CONSEC']}",
         transform=plt.gca().transAxes, fontsize=9, alpha=0.8)
if best_lock is not None:
    plt.axvline(x=best_lock, color="purple", linestyle="--", linewidth=2, label=f"Lock-in step = {best_lock}")
plt.title("Best-universe entropy evolution (chosen from MC)")
plt.xlabel("Time step"); plt.ylabel("Entropy"); plt.legend(ncol=2)
plt.grid(True, alpha=0.3)
savefig(os.path.join(FIG_DIR, "best_universe_entropy_evolution.png"))

# ======================================================
# 11) XAI (SHAP + LIME) — save PNGs and CSVs robustly
# ======================================================
try:
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.metrics import r2_score, accuracy_score
except Exception as e:
    print(f"[XAI] scikit-learn not available: {e}")
    RUN_XAI = False

# Prepare holders for metrics so they always exist
cls_acc = None
reg_r2  = None

if RUN_XAI:
    # -------- Prepare features/targets --------
    X_feat = df[["E", "I", "X"]].copy()
    y_cls  = df["stable"].astype(int).values

    reg_mask = df["lock_epoch"] >= 0
    X_reg = X_feat[reg_mask]
    y_reg = df.loc[reg_mask, "lock_epoch"].values

    # -------- Classification guard --------
    uniq_vals, uniq_cnts = np.unique(y_cls, return_counts=True)
    have_two_classes = (len(uniq_vals) == 2 and uniq_cnts.min() >= 2)
    if not have_two_classes:
        print(f"[XAI] Skipping classification: class counts = {dict(zip(uniq_vals, uniq_cnts))}")

    # -------- Regression guard --------
    have_reg = len(X_reg) >= 30
    if not have_reg:
        print(f"[XAI] Skipping regression: not enough lock-ins (have {len(X_reg)}, need >= 30).")

    # -------- Train/test splits --------
    if have_two_classes:
        Xtr_c, Xte_c, ytr_c, yte_c = train_test_split(
            X_feat, y_cls, test_size=0.25, random_state=42, stratify=y_cls
        )
        rf_cls = RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1)
        rf_cls.fit(Xtr_c, ytr_c)
        cls_acc = accuracy_score(yte_c, rf_cls.predict(Xte_c))
        print(f"[XAI] Classification accuracy (stable): {cls_acc:.3f}")
    else:
        rf_cls, Xte_c = None, None

    if have_reg:
        Xtr_r, Xte_r, ytr_r, yte_r = train_test_split(
            X_reg, y_reg, test_size=0.25, random_state=42
        )
        rf_reg = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)
        rf_reg.fit(Xtr_r, ytr_r)
        reg_r2 = r2_score(yte_r, rf_reg.predict(Xte_r))
        print(f"[XAI] Regression R^2 (lock_epoch): {reg_r2:.3f}")
    else:
        rf_reg, Xte_r = None, None

    # -------- SHAP: classification --------
    if rf_cls is not None:
        try:
            # Use TreeExplainer; fallback to generic Explainer if needed
            try:
                expl_cls = shap.TreeExplainer(rf_cls, feature_perturbation="interventional", model_output="raw")
                sv_cls = expl_cls.shap_values(Xte_c, check_additivity=False)
            except Exception:
                expl_cls = shap.Explainer(rf_cls, Xtr_c)
                sv_cls = expl_cls(Xte_c).values

            # Pick positive class if list returned
            if isinstance(sv_cls, list):
                sv_cls = sv_cls[1]
            sv_cls = np.asarray(sv_cls)
            if sv_cls.ndim == 3 and sv_cls.shape[0] == Xte_c.shape[0]:
                sv_cls = sv_cls[:, :, 1]
            elif sv_cls.ndim == 3 and sv_cls.shape[-1] == Xte_c.shape[1]:
                sv_cls = sv_cls[1, :, :]

            plt.figure()
            shap.summary_plot(sv_cls, Xte_c.values, feature_names=Xte_c.columns.tolist(), show=False)
            plt.title("SHAP summary – classification (stable)")
            plt.savefig(os.path.join(FIG_DIR, "shap_summary_cls_stable.png"), dpi=220, bbox_inches="tight")
            plt.close()

            # Save CSVs
            pd.DataFrame(sv_cls, columns=Xte_c.columns).to_csv(
                os.path.join(SAVE_DIR, "shap_values_classification.csv"), index=False
            )
            pd.Series(np.mean(np.abs(sv_cls), axis=0), index=Xte_c.columns).sort_values(ascending=False).to_csv(
                os.path.join(SAVE_DIR, "shap_feature_importance_classification.csv"),
                header=["mean_|shap|"]
            )
        except Exception as e:
            print(f"[XAI] SHAP classification failed: {e}")

    # -------- SHAP: regression --------
    if rf_reg is not None:
        try:
            try:
                expl_reg = shap.TreeExplainer(rf_reg, feature_perturbation="interventional", model_output="raw")
                sv_reg = expl_reg.shap_values(Xte_r, check_additivity=False)
            except Exception:
                expl_reg = shap.Explainer(rf_reg, Xtr_r)
                sv_reg = expl_reg(Xte_r).values

            sv_reg = np.asarray(sv_reg)
            if sv_reg.ndim == 3 and sv_reg.shape[0] == Xte_r.shape[0]:
                sv_reg = sv_reg[:, :, 0]
            elif sv_reg.ndim == 3 and sv_reg.shape[-1] == Xte_r.shape[1]:
                sv_reg = sv_reg[0, :, :]

            plt.figure()
            shap.summary_plot(sv_reg, Xte_r.values, feature_names=Xte_r.columns.tolist(), show=False)
            plt.title("SHAP summary – regression (lock_epoch)")
            plt.savefig(os.path.join(FIG_DIR, "shap_summary_reg_lock_at.png"), dpi=220, bbox_inches="tight")
            plt.close()

            pd.DataFrame(sv_reg, columns=Xte_r.columns).to_csv(
                os.path.join(SAVE_DIR, "shap_values_regression.csv"), index=False
            )
            pd.Series(np.mean(np.abs(sv_reg), axis=0), index=Xte_r.columns).sort_values(ascending=False).to_csv(
                os.path.join(SAVE_DIR, "shap_feature_importance_regression.csv"),
                header=["mean_|shap|"]
            )
        except Exception as e:
            print(f"[XAI] SHAP regression failed: {e}")

    # -------- LIME (only if classification ran) --------
    if rf_cls is not None:
        try:
            lime_explainer = LimeTabularExplainer(
                training_data=X_feat.values,
                feature_names=X_feat.columns.tolist(),
                discretize_continuous=True,
                mode='classification'
            )
            exp = lime_explainer.explain_instance(
                Xte_c.iloc[0].values, rf_cls.predict_proba, num_features=min(5, X_feat.shape[1])
            )
            lime_list = exp.as_list(label=1 if 1 in np.unique(y_cls) else 0)
            pd.DataFrame(lime_list, columns=["feature", "weight"]).to_csv(
                os.path.join(SAVE_DIR, "lime_example_classification.csv"), index=False
            )
            # Save a quick LIME bar plot as PNG
            try:
                fig = exp.as_pyplot_figure()
                fig.savefig(os.path.join(FIG_DIR, "lime_example_classification.png"), dpi=220, bbox_inches="tight")
                plt.close(fig)
            except Exception:
                pass
        except Exception as e:
            print(f"[XAI] LIME failed: {e}")

# ======================================================
# 11) Summary JSON
# ======================================================

xai_summary = {
    "did_classification": bool(RUN_XAI and ('rf_cls' in locals()) and (rf_cls is not None)),
    "did_regression":     bool(RUN_XAI and ('rf_reg' in locals()) and (rf_reg is not None)),
    "cls_accuracy":       None if cls_acc is None else float(cls_acc),
    "reg_r2":             None if reg_r2  is None else float(reg_r2),
},
summary = {
    "seeds": {"master_seed": master_seed, "universe_seeds_csv": "universe_seeds.csv"},
    "simulation": {
        "total_universes": NUM_UNIVERSES,
        "stable_fraction": float(np.mean(stables)),
        "unstable_fraction": 1.0 - float(np.mean(stables))
        "xai": xai_summary
    },
    "superposition": {
        "mean_entropy": float(np.mean(S)),
        "mean_purity": float(np.mean(P))
    },
    "collapse": {
        "E0": float(E0), "I0": float(I0), "f0": float(f0),
        "mean_X": float(np.mean(X_vals)), "std_X": float(np.std(X_vals))
    },
    "law_lockin": {
        "mean_lock_epoch": float(np.mean(valid_epochs)) if valid_epochs else None,
        "median_lock_epoch": float(np.median(valid_epochs)) if valid_epochs else None,
        "locked_fraction": float(len(valid_epochs) / len(law_epochs)) if law_epochs else 0.0,
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

print("\n✅ DONE.")
print(f"☁️ All results saved to: {SAVE_DIR}")
