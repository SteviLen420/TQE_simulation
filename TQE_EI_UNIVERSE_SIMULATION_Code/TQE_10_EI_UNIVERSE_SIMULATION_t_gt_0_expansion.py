# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Stefan Len

# ===================================================================================
# TQE_10_EI_UNIVERSE_SIMULATION_t_gt_0_expansion.py
# ===================================================================================
# Author: Stefan Len
# ===================================================================================
#
# SUMMARY:
# This script simulates the t > 0 expansion phase, modeling the evolution of each
# universe *after* its fundamental laws have locked in during the collapse stage.
#
# It models the expansion of a "size" parameter (S) over time as a multiplicative
# stochastic process. The growth at each time step is determined by two factors:
# 1.  A deterministic growth rate that is unique to each universe and is a
#     function of its initial coupled Energy-Information state (X). This
#     directly links the conditions at t=0 to the universe's subsequent fate.
# 2.  A stochastic noise term that decays over time, making the expansion
#     progressively smoother and more predictable.
#
# The script takes the final state from the collapse stage as its initial
# condition (S0). It then simulates the expansion trajectory for each universe
# independently, using a unique random seed to ensure statistical validity.
# The outputs include a .csv file with the final size of each universe, plots of
# the expansion dynamics, and a .json summary.
#
# ===================================================================================

from typing import Dict, Optional
import os, json, math, pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Cached config + paths (stabil run_id egy teljes futáson belül)
from TQE_03_EI_UNIVERSE_SIMULATION_imports import ACTIVE, PATHS, RUN_DIR, FIG_DIR

# Seeding utilities (a te fájlnevedhez igazítva)
from TQE_04_EI_UNIVERSE_SIMULATION_seeding import (
    load_or_create_run_seeds,
    universe_rngs,
)


# ---------------------------
# Helpers
# ---------------------------
def _growth_from_X(X: np.ndarray, cfg: dict) -> np.ndarray:
    """
    Map X to an effective per-step growth-rate offset (bounded / saturated).

    g_eff(X) = log(growth_base) * ( 1 + gamma * X/(1+X) ) - penalty_outside

    - gamma: X érzékenység (nagy X csak aszimptotikusan növel)
    - penalty_outside: Goldilocks ablakon kívül kis, log-skálán arányos büntetés
    """
    base = float(cfg["EXPANSION"].get("growth_base", 1.005))  # >1 ⇒ növekedés
    gamma = float(cfg["EXPANSION"].get("gamma", 1.0))
    logb = math.log(base)

    # Heurisztikus Goldilocks-penalty X-re (medián körüli ablak)
    gcfg = cfg["GOLDILOCKS"]
    center = float(np.median(X))
    iqr = float(np.quantile(X, 0.75) - np.quantile(X, 0.25))
    width = float(2.0 * (iqr + 1e-9))
    half = 0.5 * width
    inside = (X >= center - half) & (X <= center + half)

    penalty = float(gcfg.get("outside_penalty", 5.0))
    # kis log-skálás levonás, csak kívül
    penalty_out = (~inside).astype(float) * (logb * 0.25 * min(1.0, penalty / 10.0))

    xsat = X / (1.0 + X)  # (0,1)
    g_eff = logb * (1.0 + gamma * xsat) - penalty_out
    return g_eff.astype(float)


def _noise_schedule(T: int, cfg: dict) -> np.ndarray:
    """Exponential-decay zajszint padlóval, vektor hossza T."""
    ncfg = cfg["NOISE"]
    sigma0 = float(ncfg.get("exp_noise_base", 0.12))
    tau = float(ncfg.get("decay_tau", 500))
    floorf = float(ncfg.get("floor_frac", 0.25))
    ll = float(ncfg.get("ll_base_noise", 8e-4))
    t = np.arange(T, dtype=float)
    sigma = sigma0 * (floorf + (1.0 - floorf) * np.exp(-t / tau))
    sigma = np.maximum(sigma, ll)
    return sigma


def _simulate_expansion_traj(
    S0_i: float, g_eff_i: float, sigmaT: np.ndarray, dt: float, T: int, seed: int
) -> np.ndarray:
    """Deterministic expansion trajectory given a fixed seed. Returns S (length T)."""
    rng = np.random.default_rng(int(seed))
    S = np.empty(T, dtype=float)
    S[0] = max(1e-12, float(S0_i))
    for t in range(1, T):
        eps = rng.normal(0.0, sigmaT[t])
        S[t] = max(1e-18, S[t - 1] * math.exp(dt * g_eff_i + eps))
    return S


# ---------------------------
# Public API
# ---------------------------
def run_expansion(
    active_cfg: Dict = ACTIVE,
    collapse_df: Optional[pd.DataFrame] = None,
    arrays: Optional[Dict[str, np.ndarray]] = None,
):
    """
    Expand post-collapse with multiplicative growth and decaying noise.

    Optional inputs:
      - collapse_df: kimenet a run_collapse-ból (S0 = 'final_L', X ha van).
      - arrays: felülbírálható bejövők: 'S0' és/vagy 'X'.
    """
    if not active_cfg["PIPELINE"].get("run_expansion", True):
        print("[EXPANSION] run_expansion=False → skipping.")
        return {}

    # Használjuk a cache-elt útvonalakat, hogy a run_id VÁLTOZATLAN legyen
    paths = PATHS
    run_dir = pathlib.Path(RUN_DIR)
    fig_dir = pathlib.Path(FIG_DIR)
    mirrors = paths["mirrors"]

    # Seeding (megosztva a többi stádium között)
    seeds = load_or_create_run_seeds(active_cfg)
    uni_seeds = seeds["universe_seeds"]
    rngs = universe_rngs(uni_seeds)

    use_I = bool(active_cfg["PIPELINE"].get("use_information", True))
    tag = "EI" if use_I else "E"

    # --- Prepare inputs ---
    N_cfg = int(active_cfg["ENERGY"]["num_universes"])

    # X forrás prioritás
    if arrays and "X" in arrays:
        X = np.asarray(arrays["X"], dtype=float)
    elif collapse_df is not None and "X" in collapse_df.columns:
        X = collapse_df["X"].to_numpy(dtype=float)
    elif arrays and "E0" in arrays:
        X = np.asarray(arrays["E0"], dtype=float)
    elif collapse_df is not None and "E0" in collapse_df.columns:
        X = collapse_df["E0"].to_numpy(dtype=float)
    else:
        # szelíd lognormál 1 körül
        X = np.exp(np.random.normal(loc=0.0, scale=0.25, size=N_cfg)).astype(float)

    # S0 forrás prioritás
    if arrays and "S0" in arrays:
        S0 = np.asarray(arrays["S0"], dtype=float)
    elif collapse_df is not None and "final_L" in collapse_df.columns:
        S0 = collapse_df["final_L"].to_numpy(dtype=float)
    else:
        S0 = X.copy()

    # Harmonizálás (ne lépjünk túl a seed-ek számán)
    N = min(N_cfg, len(X), len(S0), len(uni_seeds))
    X = X[:N]
    S0 = S0[:N]

    # Horizont és lépés
    T = int(active_cfg["ENERGY"].get("expansion_epochs", 800))
    dt = float(active_cfg.get("FLUCTUATION", {}).get("dt", 1.0))

    # Schedules
    g_eff = _growth_from_X(X, active_cfg)   # (N,)
    sigmaT = _noise_schedule(T, active_cfg) # (T,)

    # --- Panel szimuláció ---
    S_last = np.empty(N, dtype=float)
    keep_idx = np.linspace(0, N - 1, num=min(N, 256), dtype=int)
    S_stack = []

    for i in range(N):
        rng = rngs[i]
        S = np.empty(T, dtype=float)
        S[0] = max(1e-12, float(S0[i]))
        for t in range(1, T):
            eps = rng.normal(0.0, sigmaT[t])
            S[t] = max(1e-18, S[t - 1] * math.exp(dt * g_eff[i] + eps))
        S_last[i] = S[-1]
        if i in keep_idx:
            S_stack.append(S)

    # --- CSV ---
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
    def _stats(x: np.ndarray):
        return {
            "min": float(np.min(x)),
            "max": float(np.max(x)),
            "mean": float(np.mean(x)),
            "std": float(np.std(x)),
            "p25": float(np.percentile(x, 25)),
            "median": float(np.median(x)),
            "p75": float(np.percentile(x, 75)),
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
        "files": {"csv": str(csv_path)},
    }
    json_path = run_dir / f"{tag}__expansion_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # ---------------------------
    # Plots
    # ---------------------------
    figs = []
    dpi = int(ACTIVE["RUNTIME"].get("matplotlib_dpi", 180))

    # 1) Átlagos görbe a mintapanelen
    if len(S_stack) > 0:
        arr = np.vstack(S_stack)   # (K, T)
        mean_curve = arr.mean(axis=0)
        plt.figure()
        plt.plot(mean_curve, linewidth=1.5, label="⟨S_t⟩ (average)")
        plt.xlabel("epoch")
        plt.ylabel("mean size/value ⟨S_t⟩")
        plt.title("t > 0 : Expansion dynamics (average panel)")
        plt.legend()
        f1 = fig_dir / f"{tag}__avg_expansion_curve.png"
        plt.tight_layout()
        plt.savefig(f1, dpi=dpi)
        plt.close()
        figs.append(str(f1))

    # 2) Referencia univerzum (a collapse-ból: legkorábbi lock-in, különben index 0)
    ref_idx = 0
    ref_lockin = None
    if collapse_df is not None and "lockin_at" in collapse_df.columns:
        lk_arr = collapse_df.get("lockin_at").to_numpy()
        if np.any(lk_arr >= 0):
            valid = lk_arr[:N]
            if np.any(valid >= 0):
                # a legkisebb lockin epoch indexe (NaN-ok kizárva)
                masked = np.where(valid >= 0, valid, np.inf)
                ref_idx = int(np.argmin(masked))
                if np.isfinite(masked[ref_idx]):
                    ref_lockin = int(valid[ref_idx])

    S_ref = _simulate_expansion_traj(
        S0_i=S0[ref_idx],
        g_eff_i=g_eff[ref_idx],
        sigmaT=sigmaT,
        dt=dt,
        T=T,
        seed=int(uni_seeds[ref_idx]),
    )

    plt.figure()
    plt.plot(S_ref, linewidth=1.5, label="S(t)")
    plt.axhline(y=float(np.mean(S_ref)), linestyle="--", label="⟨S⟩ (equilibrium-ish)")
    if ref_lockin is not None and 0 < ref_lockin < T:
        plt.axvline(x=ref_lockin, color="red", linestyle="--", label=f"Law lock-in ≈ {ref_lockin}")
    plt.xlabel("epoch")
    plt.ylabel("Parameters")
    plt.title("t > 0 : Expansion dynamics (reference universe)")
    plt.legend()
    f2 = fig_dir / f"{tag}__reference_expansion_curve.png"
    plt.tight_layout()
    plt.savefig(f2, dpi=dpi)
    plt.close()
    figs.append(str(f2))

    # --- Mirror ---
    from shutil import copy2
    fig_sub = ACTIVE["OUTPUTS"]["local"].get("fig_subdir", "figs")
    for m in mirrors or []:
        try:
            copy2(csv_path, pathlib.Path(m) / csv_path.name)
            copy2(json_path, pathlib.Path(m) / json_path.name)
            m_fig = pathlib.Path(m) / fig_sub
            m_fig.mkdir(parents=True, exist_ok=True)
            for fp in figs:
                copy2(fp, m_fig / os.path.basename(fp))
        except Exception as e:
            print(f"[WARN] mirror copy failed for {m}: {e}")

    print(f"[EXPANSION] mode={tag} → CSV/JSON/PNGs saved under:\n  {run_dir}")

    return {"csv": str(csv_path), "json": str(json_path), "plots": figs, "table": out_df}

# --------------------------------------------------------------
# Wrapper for Master Controller
# --------------------------------------------------------------
def run_expansion_stage(active=None, active_cfg=None, **kwargs):
    cfg = active if active is not None else active_cfg
    if cfg is None:
        raise ValueError("Provide 'active' or 'active_cfg'")     
    return run_expansion(active_cfg=cfg, **kwargs)  
    
if __name__ == "__main__":
    run_expansion_stage(ACTIVE)
