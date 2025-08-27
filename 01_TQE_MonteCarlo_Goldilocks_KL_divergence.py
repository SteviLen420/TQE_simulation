# =============================================================================
# Theory of the Question of Existence (TQE)
# Energy–Information Coupling Simulation
# =============================================================================
# Author: Stefan Len
# Purpose: Monte Carlo simulation with Goldilocks_KL divergence
# =============================================================================
# SUMMARY
# This notebook implements a Monte Carlo simulation pipeline that models the
# coupling between energy (E) and information (I). The information parameter I
# is normalized from the KL divergence between random quantum states (0..1).
# We analyze stabilization (“law lock-in”) on the composite variable X = E·I,
# estimating both its probability and timing. The Goldilocks zone is detected
# via spline fitting on the P(stable | X) curve.
# =============================================================================

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

import os, time, json, math, warnings, sys, subprocess, shutil
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

# --- XAI stack: SHAP + LIME only (no eli5/captum/interpret) ---
try:
    import shap
    from lime.lime_tabular import LimeTabularExplainer
except Exception:
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "shap==0.45.0", "lime==0.2.0.1", "scikit-learn==1.5.2", "-q"])
    import shap
    from lime.lime_tabular import LimeTabularExplainer

# ======================================================
# 1) Parameters
# ======================================================
# --- Parameters controlling the simulation ---
params = {
    "N_samples": 1000,    # number of universes (Monte Carlo runs)
    "N_epoch": 30,        # number of time steps (30 gives the most precise Goldilocks zone)
    "rel_eps": 0.05,      # lock-in threshold: max allowed relative change for stability
    "sigma0": 0.5,        # baseline noise amplitude
    "alpha": 1.5,         # noise growth factor toward the edges of the Goldilocks zone
    "seed":  None         # random seed for reproducibility
}

rng = np.random.default_rng(seed=params["seed"])

# Output dirs
run_id  = time.strftime("TQE_(E,I)KL_divergence_%Y%m%d_%H%M%S")
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
# 2) Information parameter (I) via KL divergence
# ======================================================
def sample_information_param(dim=8):
    psi1, psi2 = qt.rand_ket(dim), qt.rand_ket(dim)
    p1, p2 = np.abs(psi1.full().flatten())**2, np.abs(psi2.full().flatten())**2
    p1 /= p1.sum(); p2 /= p2.sum()
    eps = 1e-12
    KL = np.sum(p1 * np.log((p1+eps) / (p2+eps)))
    return KL / (1.0 + KL)   # 0 ≤ I ≤ 1

# ======================================================
# 3) Energy sampling
# ======================================================
def sample_energy_lognormal(mu=2.5, sigma=0.9):
    return float(rng.lognormal(mean=mu, sigma=sigma))

# ======================================================
# 4) Goldilocks noise function
# ======================================================
def sigma_goldilocks(X, sigma0, alpha, E_c_low, E_c_high):
    """
    Noise function:
    - If X is outside the zone → high noise (unstable)
    - If inside → noise increases as you approach the edges
    """
    if E_c_low is None or E_c_high is None:
        return sigma0
    if X < E_c_low or X > E_c_high:
        return sigma0 * 1.5
    else:
        mid = 0.5 * (E_c_low + E_c_high)
        width = 0.5 * (E_c_high - E_c_low)
        dist = abs(X - mid) / width   # 0 in center, 1 at edges
        return sigma0 * (1 + alpha * dist**2)

# ======================================================
# 5) Lock-in simulation
# ======================================================
def simulate_lock_in(X, N_epoch, rel_eps=0.02, sigma0=0.2, alpha=1.0, E_c_low=None, E_c_high=None):
    A, ns, H = rng.normal(50, 5), rng.normal(0.8, 0.05), rng.normal(0.7, 0.08)
    locked_at, consecutive = None, 0

    for n in range(1, N_epoch+1):
        sigma = sigma_goldilocks(X, sigma0, alpha, E_c_low, E_c_high)

        A_prev, ns_prev, H_prev = A, ns, H
        A  += rng.normal(0, sigma)
        ns += rng.normal(0, sigma/10)
        H  += rng.normal(0, sigma/5)

        # relative change
        delta_rel = (abs(A - A_prev)/abs(A_prev) +
                     abs(ns - ns_prev)/abs(ns_prev) +
                     abs(H - H_prev)/abs(H_prev)) / 3.0

        if delta_rel < rel_eps:         # much stricter threshold (2%)
            consecutive += 1            # count consecutive calm steps
            if consecutive >= 15 and locked_at is None:  # need at least 20 calm steps
                locked_at = n
        else:
            consecutive = 0

    stable = 1 if (locked_at is not None and locked_at <= N_epoch) else 0
    return stable, locked_at if locked_at is not None else -1

# ======================================================
# 6) Monte Carlo universes
# ======================================================
rows = []
for i in range(params["N_samples"]):
    E   = sample_energy_lognormal()
    I   = sample_information_param(dim=8)
    X   = E * I
    stable, lock_at = simulate_lock_in(X,
                                       params["N_epoch"],
                                       params["rel_eps"],
                                       params["sigma0"],
                                       params["alpha"])
    rows.append({"E":E, "I":I, "X":X, "stable":stable, "lock_at":lock_at})

df = pd.DataFrame(rows)
df.to_csv(os.path.join(SAVE_DIR, "samples.csv"), index=False)

# ======================================================
# 7) Stability curve (binned) + dynamic Goldilocks zone
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

# --- Detect Goldilocks zone around main peak ---
peak_index = np.argmax(ys)
peak_x = xs[peak_index]

half_max = ys[peak_index] * 0.5
valid_peak = xs[ys >= half_max]

if len(valid_peak) > 0:
    E_c_low, E_c_high = valid_peak.min(), valid_peak.max()
else:
    E_c_low, E_c_high = peak_x, peak_x
    print("⚠️ No clear peak zone found, defaulting to peak only.")

# --- Plot stability curve ---
plt.figure(figsize=(8,5))
plt.scatter(xx, yy, s=30, c="blue", alpha=0.7, label="bin means")
plt.plot(xs, ys, "r-", lw=2, label="spline fit")

# always draw the zone boundary lines
plt.axvline(E_c_low, color='g', ls='--', label=f"E_c_low = {E_c_low:.1f}")
plt.axvline(E_c_high, color='m', ls='--', label=f"E_c_high = {E_c_high:.1f}")

plt.xlabel("X = E·I")
plt.ylabel("P(stable)")
plt.title("Goldilocks zone: stabilization curve")
plt.legend()
savefig(os.path.join(FIG_DIR, "stability_curve.png"))

# ======================================================
# 8) Scatter E vs I
# ======================================================
plt.figure(figsize=(7,6))
plt.scatter(df["E"], df["I"], c=df["stable"], cmap="coolwarm", s=10, alpha=0.5)
plt.xlabel("Energy (E)"); plt.ylabel("Information parameter (I)")
plt.title("Universe outcomes in (E, I) space")
cbar = plt.colorbar(label="Stable=1 / Unstable=0")
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

# --- Save bar chart ---
plt.figure()
plt.bar(["Stable", "Unstable"], [stable_count, unstable_count], color=["green", "red"])
plt.title("Universe Stability Distribution")
plt.ylabel("Number of Universes")
plt.xlabel("Category")
labels = [
f"Stable ({stable_count}, {stable_count/len(df)*100:.1f}%)",
f"Unstable ({unstable_count}, {unstable_count/len(df)*100:.1f}%)"
]
plt.xticks([0, 1], labels)

savefig(os.path.join(FIG_DIR, "stability_summary.png"))

# ======================================================
# 10) Save summary
# ======================================================
summary = {
    "params": params,
    "N_samples": int(len(df)),
    "stable_count": int(df["stable"].sum()),              # number of stable universes
    "unstable_count": int((1 - df["stable"]).sum()),      # number of unstable universes
    "stable_ratio": float(df["stable"].mean()),           # fraction of stable universes
    "unstable_ratio": float(1 - df["stable"].mean()),
    "E_c_low": E_c_low,
    "E_c_high": E_c_high,
    "figures": {
        "stability_curve": os.path.join(FIG_DIR, "stability_curve.png"),
        "scatter_EI": os.path.join(FIG_DIR, "scatter_EI.png"),
        "stability_summary": os.path.join(FIG_DIR, "stability_summary.png")
    }
}
save_json(os.path.join(SAVE_DIR, "summary.json"), summary)

# Print summary to console
print("\n✅ DONE.")
print(f"Runs: {len(df)}")
print(f"Stable universes: {summary['stable_count']}")
print(f"Unstable universes: {summary['unstable_count']}")
print(f"Stability ratio: {summary['stable_ratio']:.3f}")
print(f"Goldilocks zone: {E_c_low:.1f} – {E_c_high:.1f}" if E_c_low else "No stable zone found")
print(f"📂 Directory: {SAVE_DIR}")

save_json(os.path.join(SAVE_DIR, "summary.json"), summary)

# Print summary to console
print("\n✅ DONE.")
print(f"Runs: {len(df)}")
print(f"Stable universes: {summary['stable_count']}")
print(f"Unstable universes: {summary['unstable_count']}")
print(f"Stability ratio: {summary['stable_ratio']:.3f}")
print(f"Goldilocks zone: {E_c_low:.1f} – {E_c_high:.1f}" if E_c_low else "No stable zone found")
print(f"📂 Directory: {SAVE_DIR}")

# ======================================================
# EXTRA: Seed search — find Top-5 seeds with highest stability
# (adds results into summary.json)
# ======================================================

NUM_SEEDS = 100               # how many different seeds to test
UNIVERSES_PER_SEED = 500      # how many universes per seed (200–500 is fine for quick test)

seed_scores = []

# -- save the original RNG, then reseed one by one
_old_rng = rng

for s in range(NUM_SEEDS):
    # use a local RNG for the E component of the simulation
    rng = np.random.default_rng(seed=s)

    # optional: also reseed numpy’s global RNG,
    # so that I (quantum states) becomes more deterministic as well
    try:
        np.random.seed(s)
    except Exception:
        pass

    rows_s = []
    for i in range(UNIVERSES_PER_SEED):
        E   = sample_energy_lognormal()
        I   = sample_information_param(dim=8)
        X   = E * I
        stable, lock_at = simulate_lock_in(
            X,
            params["N_epoch"],
            params["rel_eps"],
            params["sigma0"],
            params["alpha"]
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

# restore the original RNG
rng = _old_rng

# --- sort by stability and save ---
seed_scores_sorted = sorted(seed_scores, key=lambda r: r["stable_ratio"], reverse=True)

# Top-5 to console
print("\n🏆 Top-5 seeds by stability ratio")
for r in seed_scores_sorted[:5]:
    print(f"Seed {r['seed']:3d} → stability={r['stable_ratio']:.3f}  "
          f"locked_frac={r['locked_fraction']:.3f}  mean_lock_at={r['mean_lock_at']}")

# CSV export
top_csv_path = os.path.join(SAVE_DIR, "seed_search_top.csv")
pd.DataFrame(seed_scores_sorted).to_csv(top_csv_path, index=False)
print("Seed search table saved to:", top_csv_path)

# --- add to summary and re-save ---
summary["seed_search"] = {
    "num_seeds": NUM_SEEDS,
    "universes_per_seed": UNIVERSES_PER_SEED,
    "top5": seed_scores_sorted[:5],
    "csv_path": top_csv_path
}
save_json(os.path.join(SAVE_DIR, "summary.json"), summary)

# ======================================================
# 11) XAI (SHAP + LIME) 
# ======================================================

# ---------- Features and targets ----------
X_feat = df[["E", "I", "X"]].copy()
y_cls = df["stable"].astype(int).values
reg_mask = df["lock_at"] >= 0
X_reg = X_feat[reg_mask]
y_reg = df.loc[reg_mask, "lock_at"].values

# --- Sanity checks (optional) ---
assert not np.isnan(X_feat.values).any(), "NaN in X_feat!"
if len(X_reg) > 0:
    assert not np.isnan(X_reg.values).any(), "NaN in X_reg!"

# On-demand install (only if missing)
try:
    import shap
    from lime.lime_tabular import LimeTabularExplainer
except Exception:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "shap", "lime", "scikit-learn", "-q"])
    import shap
    from lime.lime_tabular import LimeTabularExplainer

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score

# ---------- Train/Test split ----------
Xtr_c, Xte_c, ytr_c, yte_c = train_test_split(
    X_feat, y_cls, test_size=0.25, random_state=42, stratify=y_cls
)
have_reg = len(X_reg) >= 30
if have_reg:
    Xtr_r, Xte_r, ytr_r, yte_r = train_test_split(
        X_reg, y_reg, test_size=0.25, random_state=42
    )

# ---------- Train models ----------
rf_cls = RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1)
rf_cls.fit(Xtr_c, ytr_c)
cls_acc = accuracy_score(yte_c, rf_cls.predict(Xte_c))
print(f"[XAI] Classification accuracy (stable): {cls_acc:.3f}")

if have_reg:
    rf_reg = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)
    rf_reg.fit(Xtr_r, ytr_r)
    reg_r2 = r2_score(yte_r, rf_reg.predict(Xte_r))
    print(f"[XAI] Regression R^2 (lock_at): {reg_r2:.3f}")
else:
    rf_reg, reg_r2 = None, None
    print("[XAI] Not enough locked samples for regression (need ~30+).")

# ---------- SHAP: global explanations (robust, fixed shape) ----------
X_plot = Xte_c.copy()  # vagy: X_feat.sample(min(3000, len(X_feat)), random_state=42)

# TreeExplainer with "raw" output, then format normalization
try:
    expl_cls = shap.TreeExplainer(
        rf_cls, feature_perturbation="interventional", model_output="raw"
    )
    sv_cls = expl_cls.shap_values(X_plot, check_additivity=False)
except Exception:
    expl_cls = shap.Explainer(rf_cls, Xtr_c)
    sv_cls = expl_cls(X_plot).values  # (n_samples, n_features) expected

if isinstance(sv_cls, list):
    sv_cls = sv_cls[1]  # positive class
sv_cls = np.asarray(sv_cls)
if sv_cls.ndim == 3 and sv_cls.shape[0] == X_plot.shape[0]:
    sv_cls = sv_cls[:, :, 1]
elif sv_cls.ndim == 3 and sv_cls.shape[-1] == X_plot.shape[1]:
    sv_cls = sv_cls[1, :, :]
assert sv_cls.shape == X_plot.shape, f"SHAP shape {sv_cls.shape} != data shape {X_plot.shape}"

plt.figure()
shap.summary_plot(sv_cls, X_plot.values, feature_names=X_plot.columns.tolist(), show=False)
plt.title("SHAP summary – classification (stable)")
plt.savefig(os.path.join(FIG_DIR, "shap_summary_cls_stable.png"), dpi=220, bbox_inches="tight")
plt.close()

# Regression SHAP (if trained)
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
    assert sv_reg.shape == X_plot_r.shape, f"SHAP shape {sv_reg.shape} != data shape {X_plot_r.shape}"

    plt.figure()
    shap.summary_plot(sv_reg, X_plot_r.values, feature_names=X_plot_r.columns.tolist(), show=False)
    plt.title("SHAP summary – regression (lock_at)")
    plt.savefig(os.path.join(FIG_DIR, "shap_summary_reg_lock_at.png"), dpi=220, bbox_inches="tight")
    plt.close()

# ---------- LIME: local explanation (classification) ----------
lime_explainer = LimeTabularExplainer(
    training_data=Xtr_c.values,
    feature_names=X_feat.columns.tolist(),
    discretize_continuous=True,
    mode='classification'
)
exp = lime_explainer.explain_instance(Xte_c.iloc[0].values, rf_cls.predict_proba, num_features=5)
lime_list = exp.as_list(label=1)
pd.DataFrame(lime_list, columns=["feature", "weight"]).to_csv(
    os.path.join(FIG_DIR, "lime_example_classification.csv"), index=False
)

# ======================================================
# 12) Save all outputs to Google Drive
# ======================================================
GOOGLE_BASE = "/content/drive/MyDrive/TQE_(E,I)_KL_divergence"
GOOGLE_DIR = os.path.join(GOOGLE_BASE, run_id)
os.makedirs(GOOGLE_DIR, exist_ok=True)

for root, dirs, files in os.walk(SAVE_DIR):
    for file in files:
        # NINCS .txt a listában
        if file.endswith((".png", ".fits", ".csv", ".json")):
            src = os.path.join(root, file)
            dst_dir = os.path.join(GOOGLE_DIR, os.path.relpath(root, SAVE_DIR))
            os.makedirs(dst_dir, exist_ok=True)
            shutil.copy2(src, dst_dir)

print(f"☁️ All results saved to Google Drive: {GOOGLE_DIR}")

# ======================================================
# DeepSeek analysis via Ollama (very detailed English report)
# Paste this block at the END of the notebook.
# Requires an accessible Ollama server with model "deepseek-r1:7b".
# ======================================================
import os, json, time, textwrap, requests
import numpy as np
import pandas as pd

# ---- 1) Configure Ollama endpoint + model
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "deepseek-r1:7b")

def _ollama_is_up():
    try:
        r = requests.get(OLLAMA_URL, timeout=2)
        return r.status_code < 500
    except Exception:
        return False

def deepseek_generate(prompt,
                      model=OLLAMA_MODEL,
                      temperature=0.2,
                      max_tokens=None,
                      stop=None,
                      system=None):
    """
    Call Ollama /api/generate with a single prompt (non-streaming).
    Returns the 'response' text or raises on HTTP error.
    """
    url = f"{OLLAMA_URL}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": float(temperature),
        }
    }
    if max_tokens is not None:
        payload["options"]["num_predict"] = int(max_tokens)
    if stop:
        payload["stop"] = stop
    if system:
        payload["system"] = system

    r = requests.post(url, json=payload, timeout=600)
    r.raise_for_status()
    data = r.json()
    # Some builds return 'response', others may use 'done' + 'response'
    return data.get("response") or data.get("message") or json.dumps(data)

# ---- 2) Build a compact-but-rich context from the simulation (robust to missing vars)
def _safe(v, default=None):
    try:
        return v
    except Exception:
        return default

_ctx = {
    "total_universes": _safe(N, None),
    "stable_count": int(_safe(stable_count, 0)),
    "unstable_count": int(_safe(unstable_count, 0)),
    "stable_ratio": float(_safe(np.mean(stables), np.nan)) if len(_safe(stables, [])) else None,
    "median_lock_epoch": float(_safe(np.median([e for e in _safe(law_epochs, []) if e >= 0]), np.nan)),
    "seed_top5": None,
    "features_head": None,
    "summary_json": None,
}

# Try to include Top-5 seeds if computed earlier
try:
    _ctx["seed_top5"] = seed_scores_sorted[:5]
except Exception:
    _ctx["seed_top5"] = None

# Feature snapshot
try:
    _ctx["features_head"] = df[["E","I","X"]].head(10).to_dict(orient="records")
except Exception:
    _ctx["features_head"] = None

# Summary.json content (if exists)
try:
    with open(os.path.join(SAVE_DIR,"summary.json"), "r") as f:
        _ctx["summary_json"] = json.load(f)
except Exception:
    _ctx["summary_json"] = None

# ---- 3) Compose the analysis prompt (asks for a deeply-structured report in English)
analysis_prompt = textwrap.dedent(f"""
You are an expert research assistant for a physics/complex-systems simulation.

We ran a Monte Carlo simulation of universe stability in the (E, I) space,
with a Goldilocks modulation and a law lock-in mechanism. The composite variable X = E·I (and variants)
was analyzed; we also estimated the stabilization probability and the timing of law lock-in.

Below is the data context from the most recent run (JSON-like):

TOTAL_UNIVERSES: {_ctx['total_universes']}
STABLE_COUNT: {_ctx['stable_count']}
UNSTABLE_COUNT: {_ctx['unstable_count']}
STABLE_RATIO: {_ctx['stable_ratio']}
MEDIAN_LOCK_EPOCH: {_ctx['median_lock_epoch']}
SEED_TOP5 (if any): {_ctx['seed_top5']}
FEATURE_SAMPLE (first rows of [E, I, X]): {_ctx['features_head']}
SUMMARY_JSON (if present): {_ctx['summary_json']}

Deliver a rigorous, publication-style analysis in English, with the following sections:

1) Executive summary (5–8 bullet points).
2) Data sanity check (note any signs of leakage, imbalance, or artifacts).
3) Stability mechanics
   - Interpret how E, I, and X likely drive stability.
   - Discuss what the observed stable ratio implies.
   - Explain the role of the Goldilocks modulation qualitatively.
4) Law lock-in dynamics
   - Interpret the reported median lock epoch and its spread.
   - Hypothesize mechanisms behind early vs late lock-in.
5) Seed sensitivity (if SEED_TOP5 provided)
   - What does the ranking suggest about variance across seeds?
   - Practical advice for reproducibility and robust reporting.
6) Model explainability (SHAP/LIME, if available from this run)
   - What feature importance patterns would you expect?
   - How would you validate explanations against simulation rules?
7) Limitations & failure modes
   - Identify modeling assumptions, simplifications, or numerical fragilities.
8) Actionable next steps
   - Concrete experiments, ablations, hyperparameter sweeps,
     diagnostics/plots, and statistical tests to run next.
9) Appendix
   - Propose a compact checklist for future runs (inputs, metrics, and artifacts to save).

Be very concrete and technical. Use short equations or pseudo-code where useful.
Do NOT include any hidden chain-of-thought or meta commentary; write only the final analysis.
""").strip()

# ---- 4) Call DeepSeek and save the markdown
analysis_text = None
if _ollama_is_up():
    try:
        analysis_text = deepseek_generate(
            analysis_prompt,
            model=OLLAMA_MODEL,
            temperature=0.25,
            max_tokens=1800,
            stop=None,
            system="You are a precise, technical research writer. Respond in clear English."
        )
        # Save to markdown
        out_md = os.path.join(SAVE_DIR, "deepseek_analysis.md")
        with open(out_md, "w", encoding="utf-8") as f:
            f.write(analysis_text)
        print(f"📝 DeepSeek analysis saved to: {out_md}")
    except Exception as e:
        print("[DeepSeek] Request failed:", repr(e))
else:
    print("⚠️ Ollama endpoint is not reachable at", OLLAMA_URL)
    print("   - If you run Colab in 'hosted' mode, localhost is your machine, not Colab.")
    print("   - Use a Local Runtime, or expose your Ollama server securely,")
    print("     or run this analysis locally (outside Colab).")

# Optionally print the first lines for a quick preview
if analysis_text:
    print("\n--- DeepSeek analysis (preview) ---\n")
    print("\n".join(analysis_text.splitlines()[:40]))
    print("\n[... truncated ...]\n")
