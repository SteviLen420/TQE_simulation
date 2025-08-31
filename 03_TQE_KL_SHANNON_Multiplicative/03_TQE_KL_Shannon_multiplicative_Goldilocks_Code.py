# =============================================================================
# Theory of the Question of Existence (TQE)
# Energy–Information Coupling Simulation — KL × Shannon (multiplicative) [PATCHED]
# =============================================================================
# Author: Stefan Len
# Purpose: Monte Carlo simulation with Goldilocks using I = f(KL, Shannon)
#          + reproducible seeds, I==0 / eps sweep, SHAP CSVs, stratify guard,
#          + extended summary, robust Drive copy, seed search kept
# =============================================================================

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

import os, time, json, warnings, sys, subprocess, shutil
import numpy as np
import matplotlib.pyplot as plt

# --- Core deps: ensure (no heavy extras) ---
def _ensure(pkg):
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

for pkg in ["qutip", "pandas", "scipy", "scikit-learn"]:
    _ensure(pkg)

import qutip as qt
import pandas as pd
from scipy.interpolate import make_interp_spline
warnings.filterwarnings("ignore")

# --- XAI stack: SHAP + LIME only ---
try:
    import shap
    from lime.lime_tabular import LimeTabularExplainer
except Exception:
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "shap==0.45.0", "lime==0.2.0.1", "scikit-learn==1.5.2", "-q"])
    import shap
    from lime.lime_tabular import LimeTabularExplainer

# ======================================================
# 1) MASTER CONTROLLER – All key settings in one place
# ======================================================
MASTER_CTRL = {
    # --- Simulation core ---
    "N_samples": 5000,
    "N_epoch": 50,
    "rel_eps": 0.05,
    "sigma0": 0.5,
    "alpha": 1.5,
    "seed": None,  # master RNG seed
    

    # --- Stability detection ---
    "lock_consecutive": 20,    # consecutive calm steps to lock
    "regression_min": 30,      # min. locked samples for regression

    # --- Train/test split ---
    "test_size": 0.25,
    "rf_n_estimators": 400,

    # --- XAI controls ---
    "enable_SHAP": True,
    "enable_LIME": True,

    # --- Outputs ---
    "save_figs": True,
    "save_json": True,
    "save_drive_copy": True
}

# --------- PATCH: master seed generation + rng ----------
if MASTER_CTRL["seed"] is None:
    MASTER_CTRL["seed"] = int(np.random.randint(0, 2**32 - 1))
rng = np.random.default_rng(seed=MASTER_CTRL["seed"])
print(f"🎲 Using random seed: {MASTER_CTRL['seed']}")

# Output dirs
run_id  = time.strftime("TQE_(E,I)_KL_SHANNON_%Y%m%d_%H%M%S")
SAVE_DIR = os.path.join(os.getcwd(), run_id)
FIG_DIR  = os.path.join(SAVE_DIR, "figs")
os.makedirs(FIG_DIR, exist_ok=True)

def savefig(path):
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()

def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

print(f"💾 Results saved in: {SAVE_DIR}")

# ======================================================
# 2) Information parameter I = g(KL, Shannon) (multiplicative fusion)
# ======================================================
def sample_information_param(dim=8):
    # Two random kets for KL, one reused for Shannon
    psi1, psi2 = qt.rand_ket(dim), qt.rand_ket(dim)
    p1 = np.abs(psi1.full().flatten())**2
    p2 = np.abs(psi2.full().flatten())**2
    p1 /= p1.sum(); p2 /= p2.sum()
    eps = 1e-12

    # KL-divergence-based info in [0,1]
    KL = np.sum(p1 * np.log((p1 + eps) / (p2 + eps)))
    I_kl = KL / (1.0 + KL)

    # Normalized Shannon entropy of psi1 in [0,1]
    H = -np.sum(p1 * np.log(p1 + eps))
    I_shannon = H / np.log(len(p1))

    # Multiplicative coupling, squashed to [0,1]
    I_raw = I_kl * I_shannon
    I = I_raw / (1.0 + I_raw)
    return float(I)

# ======================================================
# 3) Energy sampling
# ======================================================
def sample_energy_lognormal(mu=2.5, sigma=0.9):
    return float(rng.lognormal(mean=mu, sigma=sigma))

# ======================================================
# 4) Goldilocks noise function
# ======================================================
def sigma_goldilocks(X, sigma0, alpha, E_c_low, E_c_high):
    if E_c_low is None or E_c_high is None:
        return sigma0
    if X < E_c_low or X > E_c_high:
        return sigma0 * 1.5
    mid = 0.5 * (E_c_low + E_c_high)
    width = max(0.5 * (E_c_high - E_c_low), 1e-12)
    dist = abs(X - mid) / width  # 0 center, 1 edges
    return sigma0 * (1 + alpha * dist**2)

# ======================================================
# 5) Lock-in simulation
# ======================================================
def simulate_lock_in(X, N_epoch, rel_eps=0.02, sigma0=0.2, alpha=1.0, E_c_low=None, E_c_high=None):
    A, ns, H = rng.normal(50, 5), rng.normal(0.8, 0.05), rng.normal(0.7, 0.08)
    locked_at, consecutive = None, 0
    for n in range(1, N_epoch + 1):
        sigma = sigma_goldilocks(X, sigma0, alpha, E_c_low, E_c_high)
        A_prev, ns_prev, H_prev = A, ns, H
        A  += rng.normal(0, sigma)
        ns += rng.normal(0, sigma/10)
        H  += rng.normal(0, sigma/5)
        delta_rel = (abs(A - A_prev)/abs(A_prev) +
                     abs(ns - ns_prev)/abs(ns_prev) +
                     abs(H - H_prev)/abs(H_prev)) / 3.0
        if delta_rel < rel_eps:
            consecutive += 1
            if consecutive >= MASTER_CTRL["lock_consecutive"] and locked_at is None:
                locked_at = n
        else:
            consecutive = 0
    stable = 1 if (locked_at is not None and locked_at <= N_epoch) else 0
    return stable, (locked_at if locked_at is not None else -1)

# ======================================================
# 6) Monte Carlo universes
# ======================================================
rows = []

# reset RNG to master seed (deterministic universes)
rng = np.random.default_rng(seed=MASTER_CTRL["seed"])

for i in range(MASTER_CTRL["N_samples"]):
    # per-universe seed derived deterministically from master seed
    seed_val = rng.integers(0, 2**32 - 1)
    np.random.seed(seed_val)  # for libs that rely on np.random

    E = sample_energy_lognormal()
    I = sample_information_param(dim=8)
    X = E * I
    stable, lock_at = simulate_lock_in(
        X,
        MASTER_CTRL["N_epoch"],
        MASTER_CTRL["rel_eps"],
        MASTER_CTRL["sigma0"],
        MASTER_CTRL["alpha"]
    )
    rows.append({
        "E": E, "I": I, "X": X,
        "stable": stable, "lock_at": lock_at,
        "seed": int(seed_val)
    })

df = pd.DataFrame(rows)
df.to_csv(os.path.join(SAVE_DIR, "samples.csv"), index=False)

# save master + per-universe seeds
with open(os.path.join(SAVE_DIR, "master_seed.json"), "w") as f:
    json.dump({"master_seed": MASTER_CTRL["seed"]}, f, indent=2)

pd.DataFrame({"universe_id": range(len(df)), "seed": df["seed"]}).to_csv(
    os.path.join(SAVE_DIR, "universe_seeds.csv"), index=False
)

# ======================================================
# 7) Stability curve (binned) + dynamic Goldilocks window
# ======================================================
bins = np.linspace(df["X"].min(), df["X"].max(), 40)
df["bin"] = np.digitize(df["X"], bins)

bin_stats = df.groupby("bin").agg(
    mean_X=("X", "mean"),
    stable_rate=("stable", "mean"),
    count=("stable", "size")
).dropna()

xx = bin_stats["mean_X"].values
yy = bin_stats["stable_rate"].values

if len(xx) > 3:
    spline = make_interp_spline(xx, yy, k=3)
    xs = np.linspace(xx.min(), xx.max(), 300)
    ys = spline(xs)
else:
    xs, ys = xx, yy

peak_index = int(np.argmax(ys))
peak_x = float(xs[peak_index])
half_max = float(ys[peak_index] * 0.5)
valid_peak = xs[ys >= half_max]
if len(valid_peak) > 0:
    E_c_low, E_c_high = float(valid_peak.min()), float(valid_peak.max())
else:
    E_c_low, E_c_high = peak_x, peak_x
    print("⚠️ No clear peak zone found, defaulting to peak only.")

plt.figure(figsize=(8,5))
plt.scatter(xx, yy, s=30, c="blue", alpha=0.7, label="bin means")
plt.plot(xs, ys, "r-", lw=2, label="spline fit")
plt.axvline(E_c_low,  color='g', ls='--', label=f"E_c_low = {E_c_low:.2f}")
plt.axvline(E_c_high, color='m', ls='--', label=f"E_c_high = {E_c_high:.2f}")
plt.xlabel("X = E·I"); plt.ylabel("P(stable)")
plt.title("Goldilocks zone: stabilization curve (KL × Shannon)")
plt.legend()
savefig(os.path.join(FIG_DIR, "stability_curve.png"))

# ======================================================
# 8) Scatter E vs I
# ======================================================
plt.figure(figsize=(7,6))
sc = plt.scatter(df["E"], df["I"], c=df["stable"], cmap="coolwarm", s=10, alpha=0.5)
plt.xlabel("Energy (E)"); plt.ylabel("Information parameter (I: KL×Shannon)")
plt.title("Universe outcomes in (E, I) space")
plt.colorbar(sc, label="Stable=1 / Unstable=0")
savefig(os.path.join(FIG_DIR, "scatter_EI.png"))

# ======================================================
# 9) Stability summary (counts + percentages)
# ======================================================
stable_count = int(df["stable"].sum())
unstable_count = int(len(df) - stable_count)

print("\n🌌 Universe Stability Summary")
print(f"Total universes simulated: {len(df)}")
print(f"Stable universes:   {stable_count} ({stable_count/len(df)*100:.2f}%)")
print(f"Unstable universes: {unstable_count} ({unstable_count/len(df)*100:.2f}%)")

plt.figure()
plt.bar(["Stable", "Unstable"], [stable_count, unstable_count], color=["green", "red"])
plt.title("Universe Stability Distribution")
plt.ylabel("Number of Universes"); plt.xlabel("Category")
plt.xticks([0, 1], [
    f"Stable ({stable_count}, {stable_count/len(df)*100:.1f}%)",
    f"Unstable ({unstable_count}, {unstable_count/len(df)*100:.1f}%)"
])
savefig(os.path.join(FIG_DIR, "stability_summary.png"))

# ======================================================
# 10) PATCH: Stability by I (exact zero vs eps sweep)
# ======================================================
def _stability_stats(mask: pd.Series, label: str):
    total = int(mask.sum())
    stables = int(df.loc[mask, "stable"].sum())
    ratio = (stables / total) if total > 0 else float("nan")
    return {"group": label, "n": total, "stable_n": stables, "stable_ratio": ratio}

# Exact split
mask_I_eq0 = (df["I"] == 0.0)
mask_I_gt0 = (df["I"]  > 0.0)
zero_split_rows = [
    _stability_stats(mask_I_eq0, "I == 0"),
    _stability_stats(mask_I_gt0, "I > 0"),
]
zero_split_df = pd.DataFrame(zero_split_rows)
zero_split_path = os.path.join(SAVE_DIR, "stability_by_I_zero.csv")
zero_split_df.to_csv(zero_split_path, index=False)
print("\n📈 Stability by I (exact zero vs positive):")
print(zero_split_df.to_string(index=False))
if zero_split_df.loc[zero_split_df["group"] == "I == 0", "n"].iloc[0] == 0:
    print("⚠️ No exact I = 0 values in this sample; see epsilon sweep below.")

# Epsilon sweep
eps_list = [1e-12, 1e-9, 1e-6, 1e-3, 1e-2, 5e-2, 1e-1]
eps_rows = []
for eps in eps_list:
    eps_rows.append({**_stability_stats(df["I"] <= eps, f"I <= {eps}"), "eps": eps})
    eps_rows.append({**_stability_stats(df["I"]  > eps, f"I > {eps}"),  "eps": eps})
eps_df = pd.DataFrame(eps_rows)
eps_path = os.path.join(SAVE_DIR, "stability_by_I_eps_sweep.csv")
eps_df.to_csv(eps_path, index=False)
print("\n📈 Epsilon sweep (near-zero thresholds, preview):")
print(eps_df.head(12).to_string(index=False))
print(f"\n📝 Saved breakdowns to:\n - {zero_split_path}\n - {eps_path}")

# ======================================================
# 11) Save summary (PATCH: more fields)
# ======================================================
summary = {
    "params": MASTER_CTRL,
    "N_samples": int(len(df)),
    "stable_count": stable_count,
    "unstable_count": unstable_count,
    "stable_ratio": float(df["stable"].mean()),
    "unstable_ratio": float(1.0 - df["stable"].mean()),
    "E_c_low": E_c_low,
    "E_c_high": E_c_high,
    "seed": MASTER_CTRL["seed"],
    "master_seed": MASTER_CTRL["seed"],
    "figures": {
        "stability_curve": os.path.join(FIG_DIR, "stability_curve.png"),
        "scatter_EI": os.path.join(FIG_DIR, "scatter_EI.png"),
        "stability_summary": os.path.join(FIG_DIR, "stability_summary.png")
    }
}
save_json(os.path.join(SAVE_DIR, "summary.json"), summary)

print("\n✅ DONE (core).")
print(f"Runs: {len(df)}")
print(f"Stable universes: {summary['stable_count']}")
print(f"Unstable universes: {summary['unstable_count']}")
print(f"Stability ratio: {summary['stable_ratio']:.3f}")
print(f"Goldilocks zone: {E_c_low:.2f} – {E_c_high:.2f}")
print(f"📂 Directory: {SAVE_DIR}")

# ======================================================
# EXTRA: Seed search — Top-5 seeds with highest stability (kept)
# ======================================================
seed_scores = []
_old_rng = rng

for s in range(MASTER_CTRL["seed_search_num"]):
    rng = np.random.default_rng(seed=s)
    try:
        np.random.seed(s)
    except Exception:
        pass

    rows_s = []
    for i in range(MASTER_CTRL["seed_search_universes"]):
        E = sample_energy_lognormal()
        I = sample_information_param(dim=8)
        X = E * I
        stable, lock_at = simulate_lock_in(
            X,
            MASTER_CTRL["N_epoch"],
            MASTER_CTRL["rel_eps"],
            MASTER_CTRL["sigma0"],
            MASTER_CTRL["alpha"]
        )
        rows_s.append({"E":E, "I":I, "X":X, "stable":stable, "lock_at":lock_at})

    df_s = pd.DataFrame(rows_s)
    ratio = float(df_s["stable"].mean())
    locked_mask = df_s["lock_at"] >= 0
    locked_frac = float(locked_mask.mean()) if len(df_s) else 0.0
    mean_lock = float(df_s.loc[locked_mask, "lock_at"].mean()) if locked_mask.any() else None

    seed_scores.append({
        "seed": s,
        "stable_ratio": ratio,
        "locked_fraction": locked_frac,
        "mean_lock_at": mean_lock
    })

rng = _old_rng
seed_scores_sorted = sorted(seed_scores, key=lambda r: r["stable_ratio"], reverse=True)

print("\n🏆 Top-5 seeds by stability ratio")
for r in seed_scores_sorted[:5]:
    print(f"Seed {r['seed']:3d} → stability={r['stable_ratio']:.3f}  "
          f"locked_frac={r['locked_fraction']:.3f}  mean_lock_at={r['mean_lock_at']}")

top_csv_path = os.path.join(SAVE_DIR, "seed_search_top.csv")
pd.DataFrame(seed_scores_sorted).to_csv(top_csv_path, index=False)
print("Seed search table saved to:", top_csv_path)

summary["seed_search"] = {
    "num_seeds": MASTER_CTRL["seed_search_num"],
    "universes_per_seed": MASTER_CTRL["seed_search_universes"],
    "top5": seed_scores_sorted[:5],
    "csv_path": top_csv_path
}
if MASTER_CTRL["save_json"]:
    save_json(os.path.join(SAVE_DIR, "summary.json"), summary)

# ======================================================
# 12) XAI (SHAP + LIME) — stratify guard + CSV saves
# ======================================================
def _savefig_safe(path):
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()

def _save_df_safe(df_in, path):
    try:
        df_in.to_csv(path, index=False)
        print(f"[SAVE] CSV: {path}")
    except Exception as e:
        print(f"[ERR] CSV save failed: {path} -> {e}")

# Features & targets
X_feat = df[["E", "I", "X"]].copy()
y_cls  = df["stable"].astype(int).values
reg_mask = df["lock_at"] >= 0
X_reg = X_feat[reg_mask]
y_reg = df.loc[reg_mask, "lock_at"].values

# Sanity checks
assert not np.isnan(X_feat.values).any(), "NaN in X_feat!"
if len(X_reg) > 0:
    assert not np.isnan(X_reg.values).any(), "NaN in X_reg!"

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score

# --------- Stratify guard ----------
vals, cnts = np.unique(y_cls, return_counts=True)
can_stratify = (len(vals) == 2) and (cnts.min() >= 2)
stratify_arg = y_cls if can_stratify else None
if not can_stratify:
    print(f"[WARN] Skipping stratify: class counts = {dict(zip(vals, cnts))}")

# Classification split
Xtr_c, Xte_c, ytr_c, yte_c = train_test_split(
    X_feat, y_cls,
    test_size=MASTER_CTRL["test_size"],
    random_state=42,
    stratify=stratify_arg
)

# Regression split
have_reg = len(X_reg) >= MASTER_CTRL["regression_min"]
if have_reg:
    Xtr_r, Xte_r, ytr_r, yte_r = train_test_split(
        X_reg, y_reg,
        test_size=MASTER_CTRL["test_size"],
        random_state=42
    )

# ---------- Train models ----------
rf_cls = RandomForestClassifier(
    n_estimators=MASTER_CTRL["rf_n_estimators"],
    random_state=42,
    n_jobs=-1
)
rf_cls.fit(Xtr_c, ytr_c)
cls_acc = accuracy_score(yte_c, rf_cls.predict(Xte_c))
print(f"[XAI] Classification accuracy (stable): {cls_acc:.3f}")

rf_reg, reg_r2 = None, None
if have_reg:
    rf_reg = RandomForestRegressor(
        n_estimators=MASTER_CTRL["rf_n_estimators"],
        random_state=42,
        n_jobs=-1
    )
    rf_reg.fit(Xtr_r, ytr_r)
    reg_r2 = r2_score(yte_r, rf_reg.predict(Xte_r))
    print(f"[XAI] Regression R^2 (lock_at): {reg_r2:.3f}")
else:
    print("[XAI] Not enough locked samples for regression (need ~30+).")

# ---------- SHAP classification ----------
if MASTER_CTRL["enable_SHAP"]:
    try:
        X_plot = Xte_c.copy()
        try:
            expl_cls = shap.TreeExplainer(rf_cls, feature_perturbation="interventional", model_output="raw")
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
        assert sv_cls.shape == X_plot.shape, f"SHAP shape mismatch: {sv_cls.shape} != {X_plot.shape}"

        plt.figure()
        shap.summary_plot(sv_cls, X_plot.values, feature_names=X_plot.columns.tolist(), show=False)
        _savefig_safe(os.path.join(FIG_DIR, "shap_summary_cls_stable.png"))

        _save_df_safe(pd.DataFrame(sv_cls, columns=X_plot.columns),
                      os.path.join(FIG_DIR, "shap_values_classification.csv"))
        np.save(os.path.join(FIG_DIR, "shap_values_cls.npy"), sv_cls)

    except Exception as e:
        print(f"[ERR] SHAP classification failed: {e}")

# ---------- SHAP regression ----------
if MASTER_CTRL["enable_SHAP"] and rf_reg is not None:
    try:
        X_plot_r = Xte_r.copy()
        try:
            expl_reg = shap.TreeExplainer(rf_reg, feature_perturbation="interventional", model_output="raw")
            sv_reg = expl_reg.shap_values(X_plot_r, check_additivity=False)
        except Exception:
            expl_reg = shap.Explainer(rf_reg, Xtr_r)
            sv_reg = expl_reg(X_plot_r).values

        sv_reg = np.asarray(sv_reg)
        if sv_reg.ndim == 3 and sv_reg.shape[0] == X_plot_r.shape[0]:
            sv_reg = sv_reg[:, :, 0]
        elif sv_reg.ndim == 3 and sv_reg.shape[-1] == X_plot_r.shape[1]:
            sv_reg = sv_reg[0, :, :]
        assert sv_reg.shape == X_plot_r.shape, f"SHAP shape mismatch: {sv_reg.shape} != {X_plot_r.shape}"

        plt.figure()
        shap.summary_plot(sv_reg, X_plot_r.values, feature_names=X_plot_r.columns.tolist(), show=False)
        _savefig_safe(os.path.join(FIG_DIR, "shap_summary_reg_lock_at.png"))

        _save_df_safe(pd.DataFrame(sv_reg, columns=X_plot_r.columns),
                      os.path.join(FIG_DIR, "shap_values_regression.csv"))
        np.save(os.path.join(FIG_DIR, "shap_values_reg.npy"), sv_reg)

    except Exception as e:
        print(f"[ERR] SHAP regression failed: {e}")

# ---------- LIME ----------
if MASTER_CTRL["enable_LIME"] and len(np.unique(y_cls)) > 1:
    try:
        lime_explainer = LimeTabularExplainer(
            training_data=Xtr_c.values,
            feature_names=X_feat.columns.tolist(),
            discretize_continuous=True,
            mode='classification'
        )
        exp = lime_explainer.explain_instance(
            Xte_c.iloc[0].values, rf_cls.predict_proba, num_features=min(5, X_feat.shape[1])
        )
        lime_list = exp.as_list(label=1)
        lime_df = pd.DataFrame(lime_list, columns=["feature", "weight"])
        _save_df_safe(lime_df, os.path.join(FIG_DIR, "lime_example_classification.csv"))

        plt.figure(figsize=(6,4))
        plt.barh(lime_df["feature"], lime_df["weight"])
        plt.xlabel("LIME weight"); plt.ylabel("Feature"); plt.title("LIME explanation (stable=1)")
        plt.tight_layout()
        _savefig_safe(os.path.join(FIG_DIR, "lime_example_classification.png"))

    except Exception as e:
        print(f"[ERR] LIME failed: {e}")

# ======================================================
# 13) PATCH: Robust copy to Google Drive (counts + .txt allowed)
# ======================================================
if MASTER_CTRL["save_drive_copy"]:
    print("\n[INFO] Files in FIG_DIR before Drive copy:")
    for fn in sorted(os.listdir(FIG_DIR)):
        print("   -", fn)

    GOOGLE_BASE = "/content/drive/MyDrive/TQE_(E,I)_KL_Shannon"
    GOOGLE_DIR = os.path.join(GOOGLE_BASE, run_id)
    os.makedirs(GOOGLE_DIR, exist_ok=True)

    copied, skipped = [], []
    for root, dirs, files in os.walk(SAVE_DIR):
        dst_dir = os.path.join(GOOGLE_DIR, os.path.relpath(root, SAVE_DIR))
        os.makedirs(dst_dir, exist_ok=True)
        for file in files:
            if not file.endswith((".png", ".fits", ".csv", ".json", ".txt", ".npy")):
                continue
            src = os.path.join(root, file)
            dst = os.path.join(dst_dir, file)
            try:
                if os.path.exists(dst) and os.path.samefile(src, dst):
                    skipped.append(dst)
                    continue
            except Exception:
                pass
            shutil.copy2(src, dst)
            copied.append(dst)

    print("☁️ Copy finished.")
    print(f"Copied: {len(copied)} files")
    print(f"Skipped (same path): {len(skipped)} files")
    print("Google Drive folder:", GOOGLE_DIR)
