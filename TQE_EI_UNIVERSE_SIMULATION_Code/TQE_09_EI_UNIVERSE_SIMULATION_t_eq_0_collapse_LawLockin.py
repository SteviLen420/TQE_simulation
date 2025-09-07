# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Stefan Len

# ===================================================================================
# TQE_09_EI_UNIVERSE_SIMULATION_t_eq_0_collapse_LawLockin.py
# ===================================================================================
# Author: Stefan Len
# ===================================================================================

from typing import Dict, Optional
import os, json, math, pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Cached config + resolved paths for the current run (stable run_id)
from TQE_03_EI_UNIVERSE_SIMULATION_imports import ACTIVE, PATHS, RUN_DIR, FIG_DIR
from TQE_04_EI_UNIVERSE_SIMULATION_seeding import load_or_create_run_seeds

# ---------------------------
# Goldilocks shaping over X
# ---------------------------
def _goldilocks_noise_scale(X: np.ndarray, cfg: dict) -> np.ndarray:
    """
    Return a multiplicative noise scale s(X).
    - Inside the Goldilocks window: shrink noise (1 / (1 + sigma_alpha))
    - Outside: boost noise (outside_penalty)
    If dynamic mode is requested but no learned window exists yet,
    fallback to a heuristic window around median(X).
    """
    g = cfg["GOLDILOCKS"]
    mode = g.get("mode", "dynamic")
    if mode == "heuristic":
        center = float(g.get("E_center", 4.0))
        width  = float(g.get("E_width", 4.0))
    else:
        center = float(np.median(X))
        iqr    = float(np.quantile(X, 0.75) - np.quantile(X, 0.25))
        width  = max(1e-12, 2.0 * (iqr if iqr > 0 else 0.5 * max(center, 1e-12)))

    half = 0.5 * width
    inside = (X >= center - half) & (X <= center + half)

    sigma_alpha     = float(g.get("sigma_alpha", 1.5))
    outside_penalty = float(g.get("outside_penalty", 5.0))

    s = np.where(inside, 1.0 / (1.0 + sigma_alpha), outside_penalty)
    return s.astype(np.float64)

# ---------------------------
# Core collapse simulation
# ---------------------------
def _simulate_law_trajectory(
    X_row: float,
    epochs: int,
    rng: np.random.Generator, 
    cfg: dict,
    gX: float = 1.0, # Goldilocks per-universe noise scale (applied to sigma0)
): 
    
    """
    Simulate a single universe's law value L_t with decaying noise and mild drift.

    Returns:
        L: (epochs,) array of law values
        rel_d: (epochs-1,) per-step relative deltas
        stable_at: int or -1
        lockin_at: int or -1
    """
    stab   = cfg["STABILITY"]
    noise  = cfg["NOISE"]

    # thresholds and gates
    rel_eps_stable = float(stab.get("rel_eps_stable", 1e-2))
    rel_eps_lockin = float(stab.get("rel_eps_lockin", 5e-3))
    calm_stable    = int(stab.get("calm_steps_stable", 10))
    calm_lockin    = int(stab.get("calm_steps_lockin", 12))
    min_lock_epoch = int(stab.get("min_lockin_epoch", 200))
    require_stable = bool(stab.get("lockin_requires_stable", True))
    min_stable_ep  = int(stab.get("lockin_min_stable_epoch", 0))

    # noise dynamics (apply Goldilocks scale directly to sigma0)
    sigma0    = float(noise.get("exp_noise_base", 0.12)) * float(gX)
    ll_floor  = float(noise.get("ll_base_noise", 8e-4))
    tau       = float(noise.get("decay_tau", 500))
    floor_frac= float(noise.get("floor_frac", 0.25))

    # X-modulation of noise (larger X → smaller noise, monotonically)
    Xn = max(0.0, float(X_row))
    xnorm = Xn / (1.0 + Xn)
    sX = 1.0 / (1.0 + 2.0 * xnorm)  # in (0,1]

    # initialize near X (bounded positive)
    L = np.empty(epochs, dtype=np.float64)
    L[0] = max(1e-9, Xn)

    rel_d = np.empty(epochs - 1, dtype=np.float64)
    stable_at = -1
    lockin_at = -1
    consec_stable = 0
    consec_lockin = 0

    for t in range(1, epochs):
        decay = math.exp(-t / tau)
        sigma_t = max(ll_floor, sigma0 * (floor_frac + (1.0 - floor_frac) * decay))
        sigma_eff = sigma_t * sX

        # small OU-like drift toward Xn
        kappa = 0.02
        drift = kappa * (Xn - L[t-1])

        eps = rng.normal(loc=0.0, scale=sigma_eff)
        L[t] = max(1e-12, L[t-1] + drift + eps)

        denom = max(1e-12, abs(L[t-1]))
        rel = abs(L[t] - L[t-1]) / denom
        rel_d[t-1] = rel

        if rel < rel_eps_stable:
            consec_stable += 1
        else:
            consec_stable = 0

        if rel < rel_eps_lockin:
            consec_lockin += 1
        else:
            consec_lockin = 0

        if (stable_at < 0) and (consec_stable >= calm_stable):
            stable_at = t

        lockin_gate_ok = (t >= min_lock_epoch)
        if require_stable:
            lockin_gate_ok = lockin_gate_ok and (stable_at >= 0) and (t >= stable_at + min_stable_ep)

        if (lockin_at < 0) and lockin_gate_ok and (consec_lockin >= calm_lockin):
            lockin_at = t

        if (lockin_at > 0) and ((t - lockin_at) > 10):
            break

    return L, rel_d, int(stable_at), int(lockin_at)

# ---------------------------
# Public API
# ---------------------------
def run_collapse(
    active_cfg: Dict = ACTIVE,
    df: Optional[pd.DataFrame] = None,
    arrays: Optional[Dict[str, np.ndarray]] = None,
):
    """
    Run law collapse/lock-in detection for a population.

    Inputs (optional):
      - df: DataFrame containing at least one of ['X','E0','E'].
      - arrays: dict with numpy arrays, e.g. {'X': ..., 'E0': ..., 'I_fused': ...}
    If neither is supplied, a synthetic X scale is generated from config.
    """
    if not active_cfg["PIPELINE"].get("run_lockin", True):
        print("[COLLAPSE] run_lockin=False → skipping.")
        return {}

    # Use cached run paths (no re-resolve, no drive mount here)
    paths  = PATHS
    run_dir = pathlib.Path(RUN_DIR)
    fig_dir = pathlib.Path(FIG_DIR)
    mirrors = paths["mirrors"]

    use_I = bool(active_cfg["PIPELINE"].get("use_information", True))
    tag = "EI" if use_I else "E"

    # --- Prepare X per universe ---
    N = int(active_cfg["ENERGY"].get("num_universes", 1000))
    if arrays and isinstance(arrays, dict):
        if "X" in arrays and arrays["X"] is not None:
            X = np.asarray(arrays["X"], dtype=float)
        elif "E0" in arrays:
            X = np.asarray(arrays["E0"], dtype=float)
        elif "E" in arrays:
            X = np.asarray(arrays["E"], dtype=float)
        else:
            X = np.maximum(1e-9, np.abs(np.random.normal(loc=1.0, scale=0.5, size=N)))
    elif df is not None and isinstance(df, pd.DataFrame):
        if "X" in df.columns:
            X = df["X"].to_numpy(dtype=float)
        elif "E0" in df.columns:
            X = df["E0"].to_numpy(dtype=float)
        elif "E" in df.columns:
            X = df["E"].to_numpy(dtype=float)
        else:
            X = np.maximum(1e-9, np.abs(np.random.normal(loc=1.0, scale=0.5, size=len(df))))
        N = len(X)
    else:
        # Generate synthetic X using the master seed for reproducibility
        seeds_data = load_or_create_run_seeds(active_cfg)
        master_seed = seeds_data["master_seed"]
        rng = np.random.default_rng(master_seed)
        mu, sig = float(active_cfg["ENERGY"]["log_mu"]), float(active_cfg["ENERGY"]["log_sigma"])
        X = rng.lognormal(mean=mu, sigma=sig, size=N).astype(np.float64)

    # Goldilocks noise shaping vector for all universes
    g_scale = _goldilocks_noise_scale(X, active_cfg)

    epochs = int(active_cfg["ENERGY"].get("lockin_epochs", 500))
    # Get master seed and per-universe seeds from the central seeder
    seeds_data = load_or_create_run_seeds(active_cfg)
    universe_seeds = seeds_data["universe_seeds"]

    # --- Simulate per-universe ---
    L_last   = np.empty(N, dtype=float)
    rel_last = np.empty(N, dtype=float)
    stable_at = np.full(N, -1, dtype=int)
    lockin_at = np.full(N, -1, dtype=int)

    want_avg    = bool(active_cfg["OUTPUTS"].get("plot_avg_lockin", True))
    max_for_avg = 256
    keep_idx    = np.linspace(0, N - 1, num=min(N, max_for_avg), dtype=int)
    L_stack     = []

    for i in range(N):
        # Use the unique seed for this specific universe
        si = int(universe_seeds[i])
        # Create a generator from the universe's unique seed
        rng_universe = np.random.default_rng(si)
        L, rel_d, st, lk = _simulate_law_trajectory(
            float(X[i]), epochs, rng_universe, active_cfg, gX=float(g_scale[i])
        )
        L_last[i]    = L[-1]
        rel_last[i]  = rel_d[-1]
        stable_at[i] = st
        lockin_at[i] = lk

        if want_avg and (i in keep_idx):
            if L.shape[0] < epochs:
                pad = np.full(epochs - L.shape[0], L[-1], dtype=float)
                Lp = np.concatenate([L, pad], axis=0)
            else:
                Lp = L[:epochs]
            L_stack.append(Lp)

    # --- Build table and save CSV ---
    out_df = pd.DataFrame({
        "universe_id": np.arange(N, dtype=int),
        "X": X,
        "goldilocks_scale": g_scale,
        "stable_at": stable_at,
        "lockin_at": lockin_at,
        "final_L": L_last,
        "final_rel_delta": rel_last,
        "locked_in": (lockin_at >= 0).astype(int),
        "stable": (stable_at >= 0).astype(int),
    })

    csv_path = run_dir / f"{tag}__collapse_lockin.csv"
    out_df.to_csv(csv_path, index=False)

    # --- Summary helpers ---
    def _stat_int(x):
        x = x[x >= 0]
        if x.size == 0:
            return {"n": 0}
        return {
            "n": int(x.size),
            "min": int(np.min(x)),
            "p25": int(np.percentile(x, 25)),
            "median": int(np.median(x)),
            "p75": int(np.percentile(x, 75)),
            "max": int(np.max(x)),
            "mean": float(np.mean(x)),
            "std": float(np.std(x)),
        }

    # --- Plots ---
    figs = []
    dpi = int(ACTIVE["RUNTIME"].get("matplotlib_dpi", 180))

    if want_avg and len(L_stack) > 0:
        arr = np.vstack(L_stack)
        mean_curve = arr.mean(axis=0)
        plt.figure()
        plt.plot(mean_curve, linewidth=1.5)
        plt.xlabel("epoch")
        plt.ylabel("mean law value ⟨L_t⟩")
        plt.title("Average lock-in trajectory")
        f1 = fig_dir / f"{tag}__avg_lockin_curve.png"
        plt.tight_layout()
        plt.savefig(f1, dpi=dpi)
        plt.close()
        figs.append(str(f1))

    if active_cfg["OUTPUTS"].get("plot_lockin_hist", True):
        lk = lockin_at[lockin_at >= 0]
        plt.figure()
        if lk.size > 0:
            plt.hist(lk, bins=40)
        else:
            plt.text(0.5, 0.5, "No lock-ins detected", ha="center", va="center")
        plt.xlabel("lock-in epoch")
        plt.ylabel("count")
        plt.title("Lock-in epoch distribution")
        f2 = fig_dir / f"{tag}__lockin_hist.png"
        plt.tight_layout()
        plt.savefig(f2, dpi=dpi)
        plt.close()
        figs.append(str(f2))

    if active_cfg["OUTPUTS"].get("plot_stability_basic", False):
        plt.figure()
        plt.scatter(X, np.where(lockin_at >= 0, lockin_at, np.nan), s=6, alpha=0.5)
        plt.xlabel("X")
        plt.ylabel("lock-in epoch")
        plt.title("X vs lock-in epoch (nan = no lock-in)")
        f3 = fig_dir / f"{tag}__stability_basic.png"
        plt.tight_layout()
        plt.savefig(f3, dpi=dpi)
        plt.close()
        figs.append(str(f3))

    # --- Summary JSON ---
    summary = {
        "env": paths["env"],
        "run_id": paths["run_id"],
        "mode": tag,
        "N": N,
        "lockin_epochs": epochs,
        "counts": {
            "stable_n": int((stable_at >= 0).sum()),
            "lockin_n": int((lockin_at >= 0).sum()),
        },
        "stable_at": _stat_int(stable_at.copy()),
        "lockin_at": _stat_int(lockin_at.copy()),
        "files": {
            "csv": os.path.relpath(csv_path, start=run_dir),
            "json": f"{tag}__collapse_summary.json",
            "figs": [os.path.relpath(p, start=run_dir) for p in figs],
        },
    }

    json_path = run_dir / f"{tag}__collapse_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # --- Mirror copies ---
    from shutil import copy2
    for m in mirrors:
        try:
            copy2(csv_path, os.path.join(m, csv_path.name))
            copy2(json_path, os.path.join(m, json_path.name))
            fig_sub = ACTIVE["OUTPUTS"]["local"].get("fig_subdir", "figs")
            m_fig_dir = pathlib.Path(m) / fig_sub
            m_fig_dir.mkdir(parents=True, exist_ok=True)
            for fp in figs:
                copy2(fp, m_fig_dir / os.path.basename(fp))
        except Exception as e:
            print(f"[WARN] mirror copy failed for {m}: {e}")

    print(f"[COLLAPSE] mode={tag} → CSV/JSON/PNGs saved under:\n  {run_dir}")

    return {
        "csv": str(csv_path),
        "json": str(json_path),
        "plots": figs,
        "table": out_df,
    }

# Thin wrapper to match Master Control entrypoint
def run_lockin_stage(active: Dict = ACTIVE) -> Dict:
    return run_collapse(active_cfg=active)

# Allow standalone run
if __name__ == "__main__":
    run_collapse(ACTIVE)
