# ===========================================================================
# Theory of the Question of Existence (TQE) — E-only
# Vacuum fluctuation → Collapse → Expansion → Stability → Law lock-in
# ===========================================================================
# Author: Stefan Len
# Description: Energy-only (I = 0) simulation with a complete pipeline:
#   - Quantum superposition (diagnostic)
#   - Collapse snapshot at t = 0
#   - Monte Carlo over universes 
#   - Stability + law lock-in detection
#   - Optional averaged lock-in plots
#   - Full CSV/JSON/PNG outputs
#   - SHAP/LIME explainability (classification; regression if enough lock-ins)
# ===========================================================================

# ===========================================================================
# Theory of the Question of Existence (TQE) — E-only synchronized
# ===========================================================================
# Energy-only version aligned with the (E,I) pipeline
# ===========================================================================

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

import sys, subprocess, warnings
warnings.filterwarnings("ignore")

def _ensure(pkg):
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

# Required packages
for pkg in ["qutip", "pandas", "scikit-learn", "shap", "lime", "scipy", "matplotlib", "numpy"]:
    _ensure(pkg)

# ---- Imports ----
import os, time, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import qutip as qt

# ---- Save dirs ----
GOOGLE_BASE = "/content/drive/MyDrive/TQE_(E)_UNIVERSE_SIMULATION"
run_id = time.strftime("TQE_(E)_UNIVERSE_SIMULATION_%Y%m%d_%H%M%S")
SAVE_DIR = os.path.join(GOOGLE_BASE, run_id); os.makedirs(SAVE_DIR, exist_ok=True)
FIG_DIR  = os.path.join(SAVE_DIR, "figs");    os.makedirs(FIG_DIR, exist_ok=True)

def savefig(path): plt.savefig(path, dpi=180, bbox_inches="tight"); plt.close()
def savejson(path, obj): 
    with open(path, "w") as f: json.dump(obj, f, indent=2)

# ======================================================
# MASTER SIMULATION CONTROLS
# ======================================================

MASTER_CTRL = {
    # Core sizes
    "NUM_UNIVERSES":     5000,
    "TIME_STEPS":        800,
    "LOCKIN_EPOCHS":     500,
    "EXPANSION_EPOCHS":  800,

    # Energy distribution
    "E_LOG_MU":          2.5,
    "E_LOG_SIGMA":       0.8,
    "E_CENTER":          6.0,
    "E_WIDTH":           6,

    # Stability thresholds
    "REL_EPS_STABLE":    0.04,
    "REL_EPS_LOCKIN":    5e-4,
    "CALM_STEPS_STABLE": 5,
    "CALM_STEPS_LOCKIN": 5,

    # Law lock-in shaping
    "LL_TARGET_X":       5.0,
    "LL_BASE_NOISE":     1e6,

    # Expansion dynamics
    "EXP_GROWTH_BASE":   1.005,
    "EXP_NOISE_BASE":    1.0,

    # Quantum (superposition)
    "Q_NLEV":            12,
    "Q_TMAX":            10.0,
    "Q_NT":              200,
    "Q_GAMMA_BASE":      0.02,
    "Q_GAMMA_SIN":       0.01,
    "Q_GAMMA_NOISE":     0.005,
    "Q_SMALL_NOISE":     0.05,
    "Q_WINDOW":          0.5,
    "Q_WINDOW_STEPS":    5,

    # Collapse
    "COLLAPSE_TMIN":     -0.2,
    "COLLAPSE_TMAX":     0.2,
    "COLLAPSE_N":        200,
    "COLLAPSE_NOISE_PRE":0.5,
    "COLLAPSE_NOISE_POST":0.05,

    # Best-universe entropy controls
    "ENTROPY_NOISE_SCALE": 0.05,    # base fluctuation size
    "ENTROPY_NOISE_SPIKE": 0.1,    # occasional spike size
    "ENTROPY_SPIKE_PROB": 0.0001,    # spike probability per step
    "ENTROPY_SMOOTH_WIN": 25,       # smoothing window (moving avg for global entropy)

    # Best-universe deep dive
    "BEST_STEPS":        1000,
    "BEST_NUM_REGIONS":  10,
    "BEST_NUM_STATES":   500,
    "ENTROPY_STAB_THRESH": 3.5,
    "ENTROPY_CALM_EPS":  0.01,
    "ENTROPY_CALM_CONSEC": 5,
}

# ---- Master seed ----
master_seed = int(np.random.SeedSequence().generate_state(1)[0])
master_rng  = np.random.default_rng(master_seed)
print(f"[SEED] master_seed = {master_seed}")

# ======================================================
# 1) t < 0 : Quantum superposition (vacuum fluctuation)
# ======================================================
sub_seed_super = int(master_rng.integers(0, 2**32 - 1))
rng_super = np.random.default_rng(sub_seed_super)
np.random.seed(sub_seed_super)

Nlev = MASTER_CTRL["Q_NLEV"]
a = qt.destroy(Nlev)

H0 = a.dag()*a + MASTER_CTRL["Q_SMALL_NOISE"] * (rng_super.normal() * a + rng_super.normal() * a.dag())

psi0 = qt.rand_ket(Nlev)
rho0 = psi0 * psi0.dag()

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

def purity(r): return float((r*r).tr().real) if qt.isoper(r) else float((r*r.dag()).tr().real)

S = np.array([qt.entropy_vn(r) for r in states], dtype=float)
P = np.array([purity(r)       for r in states], dtype=float)

plt.figure()
plt.plot(tlist, S, label="Entropy")
plt.plot(tlist, P, label="Purity")
plt.title("t < 0 : Quantum superposition (vacuum fluctuation)")
plt.xlabel("time"); plt.legend()
savefig(os.path.join(FIG_DIR, "superposition.png"))

# ======================================================
# 2) t = 0 : Collapse (E only)
# ======================================================

sub_seed_collapse = int(master_rng.integers(0, 2**32 - 1))
rng_collapse = np.random.default_rng(sub_seed_collapse)
np.random.seed(sub_seed_collapse)

def sample_energy(rng=None, mu=None, sigma=None):
    if mu is None: mu = MASTER_CTRL["E_LOG_MU"]
    if sigma is None: sigma = MASTER_CTRL["E_LOG_SIGMA"]
    if rng is None: return float(np.random.lognormal(mean=mu, sigma=sigma))
    return float(rng.lognormal(mean=mu, sigma=sigma))

E0 = sample_energy(rng=rng_collapse)
X0 = E0  # E-only: X = E

tmin, tmax = MASTER_CTRL["COLLAPSE_TMIN"], MASTER_CTRL["COLLAPSE_TMAX"]
Npts       = MASTER_CTRL["COLLAPSE_N"]
collapse_t = np.linspace(tmin, tmax, Npts)

X_series = X0 + MASTER_CTRL["COLLAPSE_NOISE_PRE"] * rng_collapse.normal(size=Npts)
post_mask = (collapse_t >= 0)
X_series[post_mask] = X0 + MASTER_CTRL["COLLAPSE_NOISE_POST"] * rng_collapse.normal(size=post_mask.sum())

plt.figure()
plt.plot(collapse_t, X_series, "k-", alpha=0.6, label="fluctuation → lock-in")
plt.axhline(X0, color="r", ls="--", label=f"Lock-in X≈{X0:.2f}")
plt.axvline(0, color="r", lw=2)
plt.title("t = 0 : Collapse (E only)")
plt.xlabel("time (collapse)"); plt.ylabel("X = E"); plt.legend()
savefig(os.path.join(FIG_DIR, "collapse.png"))

# ======================================================
# 3) Stability criterion (E only)
# ======================================================

def f_E(E, E_c=MASTER_CTRL["E_CENTER"], sigma=MASTER_CTRL["E_WIDTH"]):
    """1D Gaussian Goldilocks for E-only."""
    return np.exp(-((E - E_c) ** 2) / (2.0 * sigma ** 2))

def is_stable(E, n_epoch=None, rel_eps=None, lock_consec=None, rng=None):
    if rng is None: rng = np.random.default_rng()
    if n_epoch is None: n_epoch = MASTER_CTRL["TIME_STEPS"]
    if rel_eps is None: rel_eps = MASTER_CTRL["REL_EPS_STABLE"]
    if lock_consec is None: lock_consec = MASTER_CTRL["CALM_STEPS_STABLE"]

    f = f_E(E)
    if f < 0.15:  # same gate as EI F_GATE_STABLE
        return 0

    X  = E * f
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
# 4) Law lock-in dynamics (E only)
# ======================================================

def law_lock_in(E, n_epoch=None, rng=None,
                f_min=0.12, target_X=None, base_noise=None, lock_eps_base=None):
    if n_epoch is None: n_epoch = MASTER_CTRL["LOCKIN_EPOCHS"]
    if rng is None: rng = np.random.default_rng()
    if target_X is None: target_X = MASTER_CTRL["LL_TARGET_X"]
    if base_noise is None: base_noise = MASTER_CTRL["LL_BASE_NOISE"]
    if lock_eps_base is None: lock_eps_base = MASTER_CTRL["REL_EPS_LOCKIN"]

    f = f_E(E)
    if f < f_min: return -1, []

    X  = E * f
    Xn = X / (1.0 + X)

    c_mean, c_sigma = 3e8, 1e7 * (1.1 - 0.3 * f)
    c_val = float(rng.normal(c_mean, c_sigma))

    history = [c_val]
    calm = 0
    locked_at = None
    lock_eps   = lock_eps_base * (1.1 - 0.5 * f)
    lock_consec = MASTER_CTRL["CALM_STEPS_LOCKIN"]

    for n in range(n_epoch):
        prev  = c_val
        shape = (1.0 + abs(X - target_X) / 10.0)
        damp  = (1.15 - 0.5 * f) * (1.05 - 0.4 * Xn)
        noise = base_noise * shape * damp * float(rng.uniform(0.8, 1.2))
        c_val = c_val + float(rng.normal(0.0, noise))

        history.append(c_val)

        delta = abs(c_val - prev) / max(abs(prev), 1e-9)
        if delta < lock_eps:
            calm += 1
            if calm >= lock_consec and locked_at is None:
                locked_at = n
        else:
            calm = 0

    return locked_at if locked_at is not None else -1, history

# ======================================================
# 5) Monte Carlo over universes (E only)
# ======================================================

E_vals, f_vals, X_vals = [], [], []
stables, law_epochs, final_cs, all_histories = [], [], [], []
sub_seeds = []

for _ in range(MASTER_CTRL["NUM_UNIVERSES"]):
    sub_seed = int(master_rng.integers(0, 2**32 - 1))
    sub_seeds.append(sub_seed)
    rng_uni = np.random.default_rng(sub_seed)

    Ei = sample_energy(rng=rng_uni)
    fi = f_E(Ei)
    Xi = Ei * fi

    E_vals.append(Ei); f_vals.append(fi); X_vals.append(Xi)

    s = is_stable(Ei, rng=rng_uni)
    stables.append(s)

    if s == 1:
        lock_epoch, c_hist = law_lock_in(Ei, rng=rng_uni)
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

print(f"\n🔒 Universes with lock-in: {len(valid_epochs)} / {MASTER_CTRL['NUM_UNIVERSES']}")

pd.DataFrame({"universe_id": np.arange(MASTER_CTRL["NUM_UNIVERSES"]), "seed": sub_seeds}).to_csv(
    os.path.join(SAVE_DIR, "universe_seeds.csv"), index=False
)

df = pd.DataFrame({
    "E": E_vals,
    "fE": f_vals,
    "X": X_vals,
    "stable": stables,
    "lock_epoch": law_epochs,
    "final_c": final_cs,
})
df.to_csv(os.path.join(SAVE_DIR, "tqe_runs.csv"), index=False)

# ======================================================
# 6) Stability summary — 3-bar chart
# ======================================================

total = MASTER_CTRL["NUM_UNIVERSES"]
stable_count   = int(np.sum(np.asarray(stables, dtype=int)))
unstable_count = max(0, total - stable_count)
lockin_count   = int(np.sum(np.asarray(law_epochs) >= 0))

p_lockin   = 100.0 * lockin_count   / total
p_stable   = 100.0 * stable_count   / total
p_unstable = 100.0 * unstable_count / total

xtick_labels = [
    f"Lock-in ({lockin_count}, {p_lockin:.1f}%)",
    f"Stable ({stable_count}, {p_stable:.1f}%)",
    f"Unstable ({unstable_count}, {p_unstable:.1f}%)",
]

yvals = [lockin_count, stable_count, unstable_count]

plt.figure()
plt.bar([0, 1, 2], yvals, color=["#6baed6", "#2ca02c", "#d62728"])
plt.xticks([0, 1, 2], xtick_labels, rotation=0)
plt.ylabel("Number of Universes")
plt.title("Universe Stability Distribution (E-only) — three categories")
plt.grid(axis="y", alpha=0.2)
plt.ylim(bottom=0)
savefig(os.path.join(FIG_DIR, "stability_three_bars.png"))

pd.DataFrame(
    {
        "metric":  ["lock_in", "stable", "unstable"],
        "count":   [lockin_count, stable_count, unstable_count],
        "percent": [p_lockin,     p_stable,     p_unstable],
        "total":   [total,        total,        total],
    }
).to_csv(os.path.join(SAVE_DIR, "stability_three_bars.csv"), index=False)

# ======================================================
# 7) Average law lock-in dynamics across universes
# ======================================================

if all_histories:
    min_len   = min(len(h) for h in all_histories)
    truncated = [h[:min_len] for h in all_histories]
    avg_c     = np.mean(truncated, axis=0)
    std_c     = np.std(truncated, axis=0)

    pd.DataFrame({"epoch": np.arange(min_len), "avg_c": avg_c, "std_c": std_c}).to_csv(
        os.path.join(SAVE_DIR, "law_lockin_avg.csv"), index=False
    )

    if median_epoch is not None:
        plt.figure()
        plt.plot(avg_c, label="Average c value")
        plt.fill_between(np.arange(min_len), avg_c-std_c, avg_c+std_c, alpha=0.3, color="blue", label="±1σ")
        plt.axvline(median_epoch, color="r", ls="--", lw=2, label=f"Median lock-in ≈ {median_epoch:.0f}")
        plt.title("Average law lock-in dynamics (E-only, Monte Carlo)")
        plt.xlabel("epoch"); plt.ylabel("c value (m/s)"); plt.legend()
        savefig(os.path.join(FIG_DIR, "law_lockin_avg.png"))

# ======================================================
# 8) t > 0 : Expansion dynamics (E only)
# ======================================================

def evolve(E, n_epoch=None, rng=None):
    if n_epoch is None: n_epoch = MASTER_CTRL["EXPANSION_EPOCHS"]
    if rng is None: rng = np.random.default_rng()
    A_series, A = [], 20.0
    for _ in range(n_epoch):
        A = A * MASTER_CTRL["EXP_GROWTH_BASE"] + float(rng.normal(0.0, MASTER_CTRL["EXP_NOISE_BASE"]))
        A_series.append(A)
    return A_series

A_series = evolve(E0, rng=master_rng)
plt.figure()
plt.plot(A_series, label="Amplitude A")
plt.axhline(np.mean(A_series), color="gray", ls="--", alpha=0.5, label="Equilibrium A")
if median_epoch is not None:
    plt.axvline(median_epoch, color="r", ls="--", lw=2, label=f"Law lock-in ≈ {int(median_epoch)}")
plt.title("t > 0 : Expansion dynamics (E only)")
plt.xlabel("epoch"); plt.ylabel("Amplitude A"); plt.legend()
savefig(os.path.join(FIG_DIR, "expansion.png"))

pd.DataFrame({"epoch": np.arange(len(A_series)), "Amplitude_A": A_series}).to_csv(
    os.path.join(SAVE_DIR, "expansion.csv"), index=False
)

# ======================================================
# 9) Histogram of lock-in epochs
# ======================================================

pd.DataFrame({"lock_epoch": valid_epochs}).to_csv(
    os.path.join(SAVE_DIR, "law_lockin_epochs.csv"), index=False
)

if len(valid_epochs) > 0:
    plt.figure()
    plt.hist(valid_epochs, bins=min(50, len(valid_epochs)), color="blue", alpha=0.7)
    if median_epoch is not None:
        plt.axvline(median_epoch, color="r", ls="--", lw=2, label=f"Median lock-in = {int(median_epoch)}")
        plt.legend()
    plt.title("Distribution of law lock-in epochs (E-only, Monte Carlo)")
    plt.xlabel("Epoch of lock-in"); plt.ylabel("Count")
    savefig(os.path.join(FIG_DIR, "law_lockin_mc.png"))

# ======================================================
# 10) Best-universe deep-dive (entropy analysis)
# ======================================================

# --- pick the "best" universe: earliest lock-in or max f(E) if none locked ---
locked_idxs = [i for i, e in enumerate(law_epochs) if e >= 0 and stables[i] == 1]
if locked_idxs:
    best_idx = locked_idxs[int(np.argmin([law_epochs[i] for i in locked_idxs]))]
    reason = f"earliest lock-in (epoch={law_epochs[best_idx]})"
else:
    best_idx = int(np.argmax(f_vals))
    reason = "no lock-ins → picked max f(E)"

from scipy.stats import entropy as _entropy

E_best = E_vals[best_idx]
print(f"[BEST] Universe index={best_idx} chosen by {reason}; E*={E_best:.3f}")

# --- entropy simulator for a single universe ---
def simulate_entropy_universe(E, 
                              steps=MASTER_CTRL["BEST_STEPS"],
                              num_regions=MASTER_CTRL["BEST_NUM_REGIONS"],
                              num_states=MASTER_CTRL["BEST_NUM_STATES"],
                              rng=None):
    if rng is None:
        rng = np.random.default_rng()

    states = np.zeros((num_regions, num_states))
    states[0, :] = 1.0  # break symmetry
    region_entropies, global_entropy = [], []
    lock_in_step, consecutive_calm = None, 0
    A, E_run = 1.0, float(E)

    for step in range(steps):
        noise_scale = max(0.02, 1.0 - step / steps)

        if step > 0:
            A = A * 1.01 + rng.normal(0, 0.02)

        E_run += rng.normal(0, 0.05)
        f_step_base = f_E(E_run)

        # --- common + individual noise ---
        base_raw = rng.normal(0, noise_scale * MASTER_CTRL["ENTROPY_NOISE_SCALE"], num_states)
        base_noise = np.convolve(base_raw, np.ones(25)/25, mode="same")

        for r in range(num_regions):
            # smaller individual noise component (30% strength)
            indiv_raw = rng.normal(0, 0.3 * noise_scale * MASTER_CTRL["ENTROPY_NOISE_SCALE"], num_states)
            indiv_noise = np.convolve(indiv_raw, np.ones(25)/25, mode="same")

            noise = base_noise + indiv_noise

            # occasional spike
            if rng.random() < MASTER_CTRL["ENTROPY_SPIKE_PROB"]:
                spike = rng.normal(0, MASTER_CTRL["ENTROPY_NOISE_SPIKE"], num_states)
                noise += np.convolve(spike, np.ones(25/25, mode="same")

            f_step = f_step_base * (1 + rng.normal(0, 0.05))
            states[r] = np.clip(states[r] + f_step * noise, 0, 1)

        # --- compute entropies ---
        region_entropies.append([_entropy(states[r]) for r in range(num_regions)])
        global_entropy.append(_entropy(states.flatten()))

        # optional smoothing (moving average)
        if MASTER_CTRL["ENTROPY_SMOOTH_WIN"] > 1 and len(global_entropy) >= MASTER_CTRL["ENTROPY_SMOOTH_WIN"]:
            win = MASTER_CTRL["ENTROPY_SMOOTH_WIN"]
            global_entropy[-1] = np.mean(global_entropy[-win:])

        # --- lock-in detection: calmness rule ---
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

# --- run simulation for best universe ---
best_region_entropies, best_global_entropy, best_lock = simulate_entropy_universe(E_best, rng=master_rng)

# --- save CSVs ---
pd.DataFrame({"time": np.arange(len(best_global_entropy)), "global_entropy": best_global_entropy}).to_csv(
    os.path.join(SAVE_DIR, "best_universe_global_entropy.csv"), index=False
)

best_re_mat = np.array(best_region_entropies)
re_cols = [f"region_{i}_entropy" for i in range(best_re_mat.shape[1])]
pd.DataFrame(best_re_mat, columns=re_cols).assign(time=np.arange(best_re_mat.shape[0])) \
    .to_csv(os.path.join(SAVE_DIR, "best_universe_region_entropies.csv"), index=False)

# --- plot results ---
plt.figure(figsize=(12, 6))
time_axis = np.arange(len(best_global_entropy))
for r in range(min(MASTER_CTRL["BEST_NUM_REGIONS"], best_re_mat.shape[1])):
    plt.plot(time_axis, best_re_mat[:, r], lw=1, alpha=0.6, label=f"Region {r} entropy")

plt.plot(time_axis, best_global_entropy, color="black", linewidth=2, label="Global entropy")
plt.axhline(y=MASTER_CTRL["ENTROPY_STAB_THRESH"], color="red", linestyle="--", label="Stability threshold")

# mark lock-in if found
if best_lock is not None:
    plt.axvline(x=best_lock, color="purple", linestyle="--", linewidth=2, label=f"Lock-in step = {best_lock}")

plt.title("Best-universe entropy evolution (E-only)")
plt.xlabel("Time step"); plt.ylabel("Entropy")
plt.legend(ncol=2); plt.grid(True, alpha=0.3)
savefig(os.path.join(FIG_DIR, "best_universe_entropy_evolution.png"))

# ======================================================
# 11) Summary JSON
# ======================================================

summary = {
    "seeds": {"master_seed": master_seed, "universe_seeds_csv": "universe_seeds.csv"},
    "simulation": {
        "total_universes": MASTER_CTRL["NUM_UNIVERSES"],
        "stable_fraction": float(np.mean(stables)),
        "unstable_fraction": 1.0 - float(np.mean(stables))
    },
    "superposition": {
        "mean_entropy": float(np.mean(S)),
        "mean_purity": float(np.mean(P))
    },
    "collapse": {
        "E0": float(E0),
        "mean_X": float(np.mean(X_vals)),
        "std_X": float(np.std(X_vals))
    },
    "law_lockin": {
        "mean_lock_epoch": float(np.mean(valid_epochs)) if valid_epochs else None,
        "median_lock_epoch": float(np.median(valid_epochs)) if valid_epochs else None,
        "locked_fraction": float(len(valid_epochs) / len(law_epochs)) if law_epochs else 0.0,
        "mean_final_c": float(np.nanmean(final_cs)) if len(final_cs) > 0 else None,
        "std_final_c": float(np.nanstd(final_cs)) if len(final_cs) > 0 else None
    }
}
savejson(os.path.join(SAVE_DIR, "summary.json"), summary)

print("\n✅ DONE.")
print(f"☁️ All results saved to Google Drive: {SAVE_DIR}")
