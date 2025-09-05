# ===================================================================================
# 10_TQE_(E,I)_UNIVERSE_SIMULATION_t>0_expansion.py
# ===================================================================================
# Author: Stefan Len
# ===================================================================================

from config import ACTIVE
from io_paths import resolve_output_paths, ensure_colab_drive_mounted
from seeding import load_or_create_run_seeds, universe_rngs

import os, json, math, pathlib
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------
# Helpers
# ---------------------------
def _growth_from_X(X: np.ndarray, cfg: dict) -> np.ndarray:
    """
    Map X to an effective per-step growth-rate offset.
    Saturated mapping so huge X doesn't explode:
        g_eff(X) = log(growth_base) * ( 1 + gamma * X / (1 + X) ) - penalty_outside
    """
    base = float(cfg["EXPANSION"].get("growth_base", 1.005))  # multiplicative base (>1 ⇒ growth)
    gamma = float(cfg["EXPANSION"].get("gamma", 1.0))         # X sensitivity
    logb = math.log(base)

    # Goldilocks penalty (simple heuristic around median(X))
    gcfg = cfg["GOLDILOCKS"]
    center = float(np.median(X))
    width  = float(2.0 * (np.quantile(X, 0.75) - np.quantile(X, 0.25) + 1e-9))
    half   = 0.5 * width
    inside = (X >= center - half) & (X <= center + half)
    penalty = float(gcfg.get("outside_penalty", 5.0))
    penalty_out = (~inside).astype(float) * (logb * 0.25 * min(1.0, penalty / 10.0))

    xsat = X / (1.0 + X)            # in (0,1)
    g_eff = logb * (1.0 + gamma * xsat) - penalty_out
    return g_eff.astype(float)


def _noise_schedule(T: int, cfg: dict) -> np.ndarray:
    """Exponential decay of noise std toward a floor (vector of length T)."""
    ncfg = cfg["NOISE"]
    sigma0 = float(ncfg.get("exp_noise_base", 0.12))
    tau    = float(ncfg.get("decay_tau", 500))
    floorf = float(ncfg.get("floor_frac", 0.25))
    ll     = float(ncfg.get("ll_base_noise", 8e-4))
    t = np.arange(T, dtype=float)
    sigma = sigma0 * (floorf + (1.0 - floorf) * np.exp(-t / tau))
    sigma = np.maximum(sigma, ll)
    return sigma


def _simulate_expansion_traj(S0_i: float, g_eff_i: float, sigmaT: np.ndarray,
                             dt: float, T: int, seed: int) -> np.ndarray:
    """
    Deterministically simulate a single expansion trajectory with a given seed.
    Returns S (length T).
    """
    rng = np.random.default_rng(int(seed))
    S = np.empty(T, dtype=float)
    S[0] = max(1e-12, float(S0_i))
    for t in range(1, T):
        eps = rng.normal(0.0, sigmaT[t])
        S[t] = max(1e-18, S[t-1] * math.exp(dt * g_eff_i + eps))
    return S


# ---------------------------
# Public API
# ---------------------------
def run_expansion(active_cfg: Dict = ACTIVE,
                  collapse_df: Optional[pd.DataFrame] = None,
                  arrays: Optional[Dict[str, np.ndarray]] = None):
    """
    Expand post-collapse with multiplicative growth and decaying noise.

    Inputs (optional):
      - collapse_df: table from run_collapse(...) (uses 'final_L' as S0 and 'X' if present).
                     If contains 'lockin_at', we can mark the reference lock-in epoch.
      - arrays: optional dict; if contains 'S0' and/or 'X', they override defaults.

    Returns:
      dict(csv, json, plots, table)
    """
    if not active_cfg["PIPELINE"].get("run_expansion", True):
        print("[EXPANSION] run_expansion=False → skipping.")
        return {}

    # Resolve output paths and seeds
    ensure_colab_drive_mounted(active_cfg)
    paths = resolve_output_paths(active_cfg)
    run_dir = pathlib.Path(paths["primary_run_dir"])
    fig_dir = pathlib.Path(paths["fig_dir"])
    mirrors = paths["mirrors"]

    seeds = load_or_create_run_seeds(active_cfg)  # master + per-universe seeds
    uni_seeds = seeds["universe_seeds"]
    rngs  = universe_rngs(uni_seeds)

    use_I = bool(active_cfg["PIPELINE"].get("use_information", True))
    tag = "EI" if use_I else "E"

    # --- Prepare inputs ---
    N = int(active_cfg["ENERGY"]["num_universes"])

    # X sourcing priority: arrays['X'] / collapse_df['X'] / arrays['E0'] / collapse_df['E0'] / random
    if arrays and "X" in arrays:
        X = np.asarray(arrays["X"], dtype=float)
    elif collapse_df is not None and "X" in collapse_df.columns:
        X = collapse_df["X"].to_numpy(dtype=float)
    elif arrays and "E0" in arrays:
        X = np.asarray(arrays["E0"], dtype=float)
    elif collapse_df is not None and "E0" in collapse_df.columns:
        X = collapse_df["E0"].to_numpy(dtype=float)
    else:
        # fallback: mild lognormal around 1
        X = np.exp(np.random.normal(loc=0.0, scale=0.25, size=N)).astype(float)

    # S0 sourcing priority: arrays['S0'] / collapse_df['final_L'] / fallback S0 ~ X
    if arrays and "S0" in arrays:
        S0 = np.asarray(arrays["S0"], dtype=float)
    elif collapse_df is not None and "final_L" in collapse_df.columns:
        S0 = collapse_df["final_L"].to_numpy(dtype=float)
    else:
        S0 = X.copy()

    # Harmonize lengths
    N = min(int(active_cfg["ENERGY"]["num_universes"]), len(X), len(S0), len(uni_seeds))
    X  = X[:N]
    S0 = S0[:N]

    # Simulation horizon and step
    T  = int(active_cfg["ENERGY"].get("expansion_epochs", 800))
    dt = float(active_cfg.get("FLUCTUATION", {}).get("dt", 1.0))

    # Schedules
    g_eff  = _growth_from_X(X, active_cfg)   # per-universe drift (length N)
    sigmaT = _noise_schedule(T, active_cfg)  # time-decaying noise (length T)

    # --- Simulate panel ---
    S_last = np.empty(N, dtype=float)
    keep_idx = np.linspace(0, N - 1, num=min(N, 256), dtype=int)  # small panel for averaging
    S_stack = []

    for i in range(N):
        rng = rngs[i]
        S = np.empty(T, dtype=float)
        S[0] = max(1e-12, float(S0[i]))
        for t in range(1, T):
            eps = rng.normal(0.0, sigmaT[t])
            S[t] = max(1e-18, S[t-1] * math.exp(dt * g_eff[i] + eps))
        S_last[i] = S[-1]
        if i in keep_idx:
            S_stack.append(S)

    # --- Save CSV ---
    out_df = pd.DataFrame({
        "universe_id": np.arange(N, dtype=int),
        "X": X,
        "S0": S0,
        "S_final": S_last,
        "growth_rate_eff": g_eff,
    })
    csv_path = run_dir / f"{tag}__expansion.csv"
    out_df.to_csv(csv_path, index=False)

    # --- Summary JSON ---
    def _stats(x):
        return {
            "min": float(np.min(x)), "max": float(np.max(x)),
            "mean": float(np.mean(x)), "std": float(np.std(x)),
            "p25": float(np.percentile(x, 25)), "median": float(np.median(x)),
            "p75": float(np.percentile(x, 75))
        }

    summary = {
        "env": paths["env"],
        "run_id": paths["run_id"],
        "mode": tag,
        "N": N,
        "epochs": T,
        "S0": _stats(S0),
        "S_final": _stats(S_last),
        "notes": "Multiplicative growth with decaying noise; X-modulated drift.",
        "files": {"csv": str(csv_path)}
    }
    json_path = run_dir / f"{tag}__expansion_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # ---------------------------
    # Plots
    # ---------------------------
    figs = []

    # 1) Average expansion curve over the panel (if any collected)
    if len(S_stack) > 0:
        arr = np.vstack(S_stack)                 # shape: (K, T)
        mean_curve = arr.mean(axis=0)            # ⟨S_t⟩
        plt.figure()
        plt.plot(mean_curve, linewidth=1.5, label="⟨S_t⟩ (average)")
        plt.xlabel("epoch")
        plt.ylabel("mean size/value ⟨S_t⟩")
        plt.title("t > 0 : Expansion dynamics (average panel)")
        plt.legend()
        f1 = fig_dir / f"{tag}__avg_expansion_curve.png"
        plt.tight_layout()
        plt.savefig(f1, dpi=ACTIVE["RUNTIME"].get("matplotlib_dpi", 180))
        plt.close()
        figs.append(str(f1))

    # 2) Reference universe curve:
    #    If collapse_df contains 'lockin_at', pick the earliest lock-in;
    #    otherwise fallback to universe 0. Mark the lock-in epoch if known.
    ref_idx = 0
    ref_lockin = None
    if collapse_df is not None and "lockin_at" in collapse_df.columns:
        lk_arr = collapse_df.get("lockin_at").to_numpy()
        if np.any(lk_arr >= 0):
            # pick earliest lock-in among the first N universes present here
            valid = lk_arr[:N]
            if np.any(valid >= 0):
                ref_idx = int(np.nanargmin(np.where(valid >= 0, valid, np.nan)))
                ref_lockin = int(valid[ref_idx])

    # Deterministic re-run for the reference universe using its own seed
    S_ref = _simulate_expansion_traj(
        S0_i=S0[ref_idx],
        g_eff_i=g_eff[ref_idx],
        sigmaT=sigmaT,
        dt=dt,
        T=T,
        seed=int(uni_seeds[ref_idx])
    )

    plt.figure()
    plt.plot(S_ref, linewidth=1.5, color="tab:blue", label="S(t)")
    plt.axhline(y=float(np.mean(S_ref)), color="gray", linestyle="--", label="⟨S⟩ (equilibrium-ish)")
    if ref_lockin is not None and ref_lockin > 0 and ref_lockin < T:
        plt.axvline(x=ref_lockin, color="red", linestyle="--", label=f"Law lock-in ≈ {ref_lockin}")
    plt.xlabel("epoch")
    plt.ylabel("Parameters")
    plt.title("t > 0 : Expansion dynamics (reference universe)")
    plt.legend()
    f2 = fig_dir / f"{tag}__reference_expansion_curve.png"
    plt.tight_layout()
    plt.savefig(f2, dpi=ACTIVE["RUNTIME"].get("matplotlib_dpi", 180))
    plt.close()
    figs.append(str(f2))

    # --- Mirror copies ---
    from shutil import copy2
    fig_sub = ACTIVE["OUTPUTS"]["local"].get("fig_subdir", "figs")
    for m in mirrors:
        try:
            copy2(csv_path, os.path.join(m, csv_path.name))
            copy2(json_path, os.path.join(m, json_path.name))
            m_fig = pathlib.Path(m) / fig_sub
            m_fig.mkdir(parents=True, exist_ok=True)
            for fp in figs:
                copy2(fp, m_fig / os.path.basename(fp))
        except Exception as e:
            print(f"[WARN] mirror copy failed for {m}: {e}")

    print(f"[EXPANSION] mode={tag} → CSV/JSON/PNGs saved under:\n  {run_dir}")

    return {"csv": str(csv_path), "json": str(json_path), "plots": figs, "table": out_df}


# Allow standalone execution
if __name__ == "__main__":
    run_expansion(ACTIVE)
