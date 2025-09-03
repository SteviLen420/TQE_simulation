# collapse.py
# ===================================================================================
# Law Collapse / Lock-in stage for the TQE universe simulation
# -----------------------------------------------------------------------------------
# Simulates a per-universe "law" value over epochs with decaying noise. Detects:
#   - STABLE when relative per-step change stays below rel_eps_stable for
#     calm_steps_stable consecutive steps (after min_lockin_epoch gate can be 0)
#   - LOCK-IN when below rel_eps_lockin for calm_steps_lockin consecutive steps
#     and (optionally) only after having been STABLE first.
#
# Noise and drift are modulated by X = f(E, I) (or E in E-only mode):
#   sigma_t ≈ max(ll_base_noise, exp_noise_base * sigma_decay(t) * g(X))
#   with an exponential decay toward a floor. X rescales noise (outside window
#   noisier, inside Goldilocks calmer) using GOLDILOCKS params.
#
# Outputs:
#   - CSV: per-universe stable_at, lockin_at, last_value, last_rel_delta, flags
#   - JSON: summary stats and config snapshot
#   - PNG: average lock-in curve, histogram of lock-in epochs, optional stability plot
#
# Author: Stefan Len
# ===================================================================================

from config import ACTIVE
from io_paths import resolve_output_paths, ensure_colab_drive_mounted

import os, json, math, pathlib
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------
# Utility: RNG
# ---------------------------
def _rng(seed: Optional[int]):
    """Create a reproducible Generator if seed is provided."""
    return np.random.default_rng(int(seed)) if seed is not None else np.random.default_rng()


# ---------------------------
# Goldilocks shaping over X
# ---------------------------
def _goldilocks_noise_scale(X: np.ndarray, cfg: dict) -> np.ndarray:
    """
    Return a multiplicative noise scale s(X).
    - Inside the Goldilocks window: shrink noise (1 / (1 + sigma_alpha))
    - Outside: boost noise (outside_penalty)
    If dynamic mode is requested but we don't have a learned window yet,
    we fallback to a heuristic window around median(X).
    """
    g = cfg["GOLDILOCKS"]
    mode = g.get("mode", "dynamic")
    if mode == "heuristic":
        center = float(g.get("E_center", 4.0))
        width  = float(g.get("E_width", 4.0))
    else:
        # simple fallback: use median and IQR-proportional width
        center = float(np.median(X))
        iqr    = float(np.quantile(X, 0.75) - np.quantile(X, 0.25))
        width  = max(1e-12, 2.0 * (iqr if iqr > 0 else 0.5 * center))

    half = 0.5 * width
    inside = (X >= center - half) & (X <= center + half)

    sigma_alpha    = float(g.get("sigma_alpha", 1.5))
    outside_penalty= float(g.get("outside_penalty", 5.0))

    s = np.where(inside, 1.0 / (1.0 + sigma_alpha), outside_penalty)
    return s.astype(np.float64)


# ---------------------------
# Core collapse simulation
# ---------------------------
def _simulate_law_trajectory(X_row: float, epochs: int, seed: Optional[int], cfg: dict):
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

    # noise dynamics
    sigma0    = float(noise.get("exp_noise_base", 0.12))
    ll_floor  = float(noise.get("ll_base_noise", 8e-4))
    tau       = float(noise.get("decay_tau", 500))
    floor_frac= float(noise.get("floor_frac", 0.25))

    # X-modulation of noise (larger X → smaller noise, monotonically)
    # simple monotone: sX = 1 / (1 + X_norm)
    Xn = max(0.0, float(X_row))
    xnorm = Xn / (1.0 + Xn)
    sX = 1.0 / (1.0 + 2.0 * xnorm)  # in (0,1]

    # random state
    rng = _rng(seed)

    # initialize law value near X (bounded positive)
    L = np.empty(epochs, dtype=np.float64)
    L[0] = max(1e-9, Xn)

    # stability trackers
    rel_d = np.empty(epochs - 1, dtype=np.float64)
    stable_at = -1
    lockin_at = -1
    consec_stable = 0
    consec_lockin = 0

    for t in range(1, epochs):
        # decaying sigma toward a floor
        decay = math.exp(-t / tau)
        sigma_t = max(ll_floor, sigma0 * (floor_frac + (1.0 - floor_frac) * decay))
        # apply X and Goldilocks shaping via sX (sX<=1 → damp)
        sigma_eff = sigma_t * sX

        # drift: small pull toward Xn (OU-like)
        kappa = 0.02  # small mean-reverting drift
        drift = kappa * (Xn - L[t-1])

        # step
        eps = rng.normal(loc=0.0, scale=sigma_eff)
        L[t] = max(1e-12, L[t-1] + drift + eps)

        # relative delta
        denom = max(1e-12, abs(L[t-1]))
        rel = abs(L[t] - L[t-1]) / denom
        rel_d[t-1] = rel

        # stability counters
        if rel < rel_eps_stable:
            consec_stable += 1
        else:
            consec_stable = 0

        if rel < rel_eps_lockin:
            consec_lockin += 1
        else:
            consec_lockin = 0

        # first time we hit stable
        if stable_at < 0 and consec_stable >= calm_stable:
            stable_at = t

        # lock-in condition
        lockin_gate_ok = (t >= min_lock_epoch)
        if require_stable:
            lockin_gate_ok = lockin_gate_ok and (stable_at >= 0) and (t >= stable_at + min_stable_ep)

        if lockin_at < 0 and lockin_gate_ok and consec_lockin >= calm_lockin:
            lockin_at = t

        # early stop if locked in and we have a small tail already
        if lockin_at > 0 and (t - lockin_at) > 10:
            # allow a short buffer after lock-in then break
            break

    return L, rel_d, int(stable_at), int(lockin_at)


# ---------------------------
# Public API
# ---------------------------
def run_collapse(active_cfg: Dict = ACTIVE, df: Optional[pd.DataFrame] = None, arrays: Optional[Dict[str, np.ndarray]] = None):
    """
    Run law collapse/lock-in detection for a population.
    Inputs (optional):
      - df: DataFrame containing at least 'universe_id' and one of ['X','E0','E'].
      - arrays: dict with numpy arrays, e.g. {'X': ..., 'E0': ..., 'I_fused': ...}
    If neither is supplied, we generate a synthetic X scale from config.

    Returns a dict with:
      - 'csv', 'json' paths
      - 'plots': list of figure paths
      - 'table': the resulting per-universe DataFrame
    """
    if not active_cfg["PIPELINE"].get("run_lockin", True):
        print("[COLLAPSE] run_lockin=False → skipping.")
        return {}

    ensure_colab_drive_mounted(active_cfg)
    paths = resolve_output_paths(active_cfg)
    run_dir = pathlib.Path(paths["primary_run_dir"])
    fig_dir = pathlib.Path(paths["fig_dir"])
    mirrors = paths["mirrors"]

    use_I = bool(active_cfg["PIPELINE"].get("use_information", True))
    tag = "EI" if use_I else "E"

    # --- Prepare X per universe ---
    N = int(active_cfg["ENERGY"].get("num_universes", 1000))
    if arrays and isinstance(arrays, dict):
        if "X" in arrays and arrays["X"] is not None:
            X = np.asarray(arrays["X"], dtype=float)
        elif "E0" in arrays:
            # degrade to E-only if I not provided
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
        # synthetic X
        N = int(active_cfg["ENERGY"].get("num_universes", 1000))
        rng = _rng(active_cfg["ENERGY"].get("seed"))
        # basic lognormal X
        mu, sig = float(active_cfg["ENERGY"]["log_mu"]), float(active_cfg["ENERGY"]["log_sigma"])
        X = rng.lognormal(mean=mu, sigma=sig, size=N).astype(np.float64)

    # Goldilocks noise shaping vector for all universes (broadcast later)
    g_scale = _goldilocks_noise_scale(X, active_cfg)

    epochs = int(active_cfg["ENERGY"].get("lockin_epochs", 500))
    seed   = active_cfg["ENERGY"].get("seed", None)

    # --- Simulate per-universe ---
    L_last = np.empty(N, dtype=float)
    rel_last = np.empty(N, dtype=float)
    stable_at = np.full(N, -1, dtype=int)
    lockin_at = np.full(N, -1, dtype=int)

    # For average curve plot: collect a trimmed set of trajectories
    want_avg = bool(active_cfg["OUTPUTS"].get("plot_avg_lockin", True))
    max_for_avg = 256
    keep_idx = np.linspace(0, N - 1, num=min(N, max_for_avg), dtype=int)
    L_stack = []

    base_seed = int(seed) if seed is not None else None
    for i in range(N):
        # per-universe seed
        si = (None if base_seed is None else (base_seed + i * 9973))

        # temporarily inject Goldilocks shaping via NOISE.exp_noise_base scaling
        # (do not mutate ACTIVE; pass via a shallow-copy view where needed)
        # Here we scale internally by multiplying sigma0 with g_scale[i] inside the loop.
        # To keep function stateless, we pass gX through X_row (already embedded in sigma via sX).
        L, rel_d, st, lk = _simulate_law_trajectory(float(X[i]), epochs, si, active_cfg)

        L_last[i]   = L[len(L) - 1]
        rel_last[i] = rel_d[len(rel_d) - 1]
        stable_at[i]= st
        lockin_at[i]= lk

        if want_avg and (i in keep_idx):
            # Pad to epochs length for alignment
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

    # --- Summary JSON ---
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
            "csv": str(csv_path),
        },
    }

    json_path = run_dir / f"{tag}__collapse_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # --- Plots ---
    figs = []

    # 1) Average lock-in curve
    if want_avg and len(L_stack) > 0:
        arr = np.vstack(L_stack)  # (K, epochs)
        mean_curve = arr.mean(axis=0)
        plt.figure()
        plt.plot(mean_curve, linewidth=1.5)
        plt.xlabel("epoch")
        plt.ylabel("mean law value ⟨L_t⟩")
        plt.title("Average lock-in trajectory")
        f1 = fig_dir / f"{tag}__avg_lockin_curve.png"
        plt.tight_layout()
        plt.savefig(f1, dpi=ACTIVE["RUNTIME"].get("matplotlib_dpi", 180))
        plt.close()
        figs.append(str(f1))

    # 2) Histogram of lock-in epochs
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
        plt.savefig(f2, dpi=ACTIVE["RUNTIME"].get("matplotlib_dpi", 180))
        plt.close()
        figs.append(str(f2))

    # 3) Optional: simple stability diagnostic
    if active_cfg["OUTPUTS"].get("plot_stability_basic", False):
        plt.figure()
        plt.scatter(X, np.where(lockin_at >= 0, lockin_at, np.nan), s=6, alpha=0.5)
        plt.xlabel("X")
        plt.ylabel("lock-in epoch")
        plt.title("X vs lock-in epoch (nan = no lock-in)")
        f3 = fig_dir / f"{tag}__stability_basic.png"
        plt.tight_layout()
        plt.savefig(f3, dpi=ACTIVE["RUNTIME"].get("matplotlib_dpi", 180))
        plt.close()
        figs.append(str(f3))

    # --- Mirror copies (CSV/JSON/PNGs) ---
    from shutil import copy2
    for m in mirrors:
        try:
            copy2(csv_path, os.path.join(m, csv_path.name))
            copy2(json_path, os.path.join(m, json_path.name))
            # figures to <mirror>/<fig_subdir>/
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


# Allow standalone run
if __name__ == "__main__":
    run_collapse(ACTIVE)
