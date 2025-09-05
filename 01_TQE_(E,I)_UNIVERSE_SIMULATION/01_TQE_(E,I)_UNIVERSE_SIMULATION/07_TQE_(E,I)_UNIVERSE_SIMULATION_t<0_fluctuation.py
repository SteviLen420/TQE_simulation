# ===================================================================================
# 07_TQE_(E,I)_UNIVERSE_SIMULATION_t<0_fluctuation.py
# ===================================================================================
# Author: Stefan Len
# ===================================================================================

from typing import Dict, Tuple, Optional
import os, json, shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import entropy

from config import ACTIVE
from io_paths import resolve_output_paths, ensure_colab_drive_mounted


# ---------------------------
# Helpers
# ---------------------------

def _rng(seed: Optional[int]):
    """Create a reproducible RNG (falls back to fresh SeedSequence if seed is None)."""
    if seed is None:
        return np.random.default_rng(np.random.SeedSequence())
    return np.random.default_rng(int(seed))

def _apply_truncation(E: np.ndarray, low, high) -> np.ndarray:
    """Clamp sampled energies to [low, high] if thresholds are provided."""
    if low is not None:
        E = np.maximum(E, float(low))
    if high is not None:
        E = np.minimum(E, float(high))
    return E

def _random_prob_vec(dim: int, rng) -> np.ndarray:
    """Draw a random probability vector by normalizing complex amplitudes."""
    a = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    p = np.abs(a) ** 2
    p /= p.sum()
    return p

def _info_components(p: np.ndarray, kl_eps: float) -> Tuple[float, float, float]:
    """
    Compute components:
      - KL(p || u) normalized by log(dim)  in [0,1]
      - H_norm = H(p)/log(dim)             in [0,1]
      - (1 - H_norm)                       as Shannon-derived "information"
    Returns: (I_kl_norm, H_norm, one_minus_H_norm)
    """
    dim = p.size
    u = np.full(dim, 1.0 / dim)

    with np.errstate(divide="ignore", invalid="ignore"):
        kl = np.sum(p * (np.log(p + kl_eps) - np.log(u + kl_eps)))
    kl_norm = float(kl / np.log(dim)) if dim > 1 else 0.0
    kl_norm = float(np.clip(kl_norm, 0.0, 1.0))

    H = entropy(p, base=np.e)
    H_norm = float(H / np.log(dim)) if dim > 1 else 0.0
    H_norm = float(np.clip(H_norm, 0.0, 1.0))

    return kl_norm, H_norm, 1.0 - H_norm

def _fuse_I(kl_norm: float, shannon_info: float, info_cfg: dict) -> float:
    """Fuse KL and Shannon components into I ∈ [0,1] according to config."""
    mode = info_cfg.get("fusion", "product")
    if mode == "product":
        val = kl_norm * shannon_info
    elif mode == "weighted":
        w_kl = float(info_cfg.get("weight_kl", 0.5))
        w_sh = float(info_cfg.get("weight_shannon", 0.5))
        s = (w_kl + w_sh) or 1.0
        val = (w_kl / s) * kl_norm + (w_sh / s) * shannon_info
    else:
        val = kl_norm * shannon_info

    exp_ = float(info_cfg.get("exponent", 1.0))
    floor = float(info_cfg.get("floor_eps", 0.0))
    val = max(val, floor)
    if exp_ != 1.0:
        val = val ** exp_
    return float(np.clip(val, 0.0, 1.0))

def _couple_X(E: np.ndarray, I: Optional[np.ndarray], x_cfg: dict) -> np.ndarray:
    """Compute X = f(E, I) (or X=E if I is None)."""
    mode   = x_cfg.get("mode", "product")
    alphaI = float(x_cfg.get("alpha_I", 0.8))
    powI   = float(x_cfg.get("I_power", 1.0))
    scale  = float(x_cfg.get("scale", 1.0))

    if I is None:
        X = E.astype(float)
    else:
        if mode == "E_plus_I":
            X = E + alphaI * I
        elif mode == "E_times_I_pow":
            X = E * (alphaI * I) ** powI
        else:
            X = E * (alphaI * I)
    return scale * X

def _save_with_mirrors(src_path: str, mirrors: list, put_in_figs: bool = False):
    """
    Copy freshly written file to mirrors.
    - CSV/JSON: put_in_figs=False  → <mirror>/<file>
    - PNG:      put_in_figs=True   → <mirror>/figs/<file>
    """
    fig_sub = ACTIVE["OUTPUTS"]["local"].get("fig_subdir", "figs")
    for m in mirrors:
        try:
            if put_in_figs:
                dst_dir = os.path.join(m, fig_sub)
                os.makedirs(dst_dir, exist_ok=True)
                dst = os.path.join(dst_dir, os.path.basename(src_path))
            else:
                dst = os.path.join(m, os.path.basename(src_path))
            shutil.copy2(src_path, dst)
        except Exception as e:
            print(f"[WARN] Mirror copy failed → {m}: {e}")


# ---------------------------
# Main stage
# ---------------------------

def run_fluctuation(active_cfg: Dict = ACTIVE, seed: Optional[int] = None) -> Dict:
    """
    Run fluctuation stage:
      1) Sample E (lognormal with optional truncation)
      2) If info channel on: compute I_kl, I_shannon(=1-H_norm), fuse to I
      3) Couple into X = f(E, I) (or X=E)
      4) Save CSV/PNGs/JSON and mirror to configured targets
    Returns a dict with arrays, dataframe, summary and path info.
    """
    # Ensure Drive (if Colab) and resolve paths
    ensure_colab_drive_mounted(active_cfg)
    paths   = resolve_output_paths(active_cfg)
    primary = paths["primary_run_dir"]
    figdir  = paths["fig_dir"]
    mirrors = paths["mirrors"]

    # EI/E filename prefix (keeps consistency with other stages)
    ei_tag_enabled = active_cfg["OUTPUTS"].get("tag_ei_in_filenames", True)
    use_I          = bool(active_cfg["PIPELINE"].get("use_information", True))
    prefix         = ("EI__" if use_I else "E__") if ei_tag_enabled else ""

    # RNG seed: explicit arg > config seed
    seed_eff = active_cfg["ENERGY"].get("seed") if seed is None else seed
    rng      = _rng(seed_eff)

    # ---------------------------
    # 1) Sample Energy (E)
    # ---------------------------
    N      = int(active_cfg["ENERGY"]["num_universes"])
    mu     = float(active_cfg["ENERGY"]["log_mu"])
    sigma  = float(active_cfg["ENERGY"]["log_sigma"])
    t_low  = active_cfg["ENERGY"].get("trunc_low", None)
    t_high = active_cfg["ENERGY"].get("trunc_high", None)

    # lognormal: draw log(E) ~ N(mu, sigma^2) → E = exp(logE)
    logE0 = rng.normal(loc=mu, scale=sigma, size=N).astype(float)
    E0    = np.exp(logE0)
    E0    = _apply_truncation(E0, t_low, t_high)

    # ---------------------------
    # 2) Optional Information (I) from random prob. vectors
    # ---------------------------
    I_kl = I_shannon = I_fused = None

    if use_I:
        info_cfg = active_cfg["INFORMATION"]
        dim      = int(info_cfg["hilbert_dim"])
        eps      = float(info_cfg["kl_eps"])

        I_kl      = np.zeros(N, dtype=float)
        H_norm    = np.zeros(N, dtype=float)   # kept for debugging/plots if needed
        I_shannon = np.zeros(N, dtype=float)   # this is (1 - H_norm)
        I_fused   = np.zeros(N, dtype=float)

        for i in range(N):
            p = _random_prob_vec(dim, rng)
            kl_norm, h_norm, sh_info = _info_components(p, eps)
            I_kl[i]      = kl_norm
            H_norm[i]    = h_norm
            I_shannon[i] = sh_info
            I_fused[i]   = _fuse_I(kl_norm, sh_info, info_cfg)

    # ---------------------------
    # 3) Coupling X = f(E, I) (or X=E)
    # ---------------------------
    X = _couple_X(E0, I_fused if use_I else None, active_cfg["COUPLING_X"])

    # ---------------------------
    # 4) Goldilocks heuristics (flag on E)
    # ---------------------------
    gcfg = active_cfg["GOLDILOCKS"]
    if gcfg.get("mode", "dynamic") == "heuristic":
        c = float(gcfg.get("E_center", 4.0))
        w = float(gcfg.get("E_width",  4.0))
    else:
        c, w = float(mu + sigma), float(2.0 * sigma)
    in_goldilocks_E = (np.abs(E0 - c) <= (w / 2.0)).astype(int)

    # ---------------------------
    # 5) DataFrame + CSV
    # ---------------------------
    data = {
        "universe_id": np.arange(N, dtype=int),
        "E0": E0,
        "logE0": logE0,
        "in_goldilocks_E": in_goldilocks_E,
        "X": X,
    }
    if use_I:
        data.update({
            "I_kl": I_kl,
            "I_shannon": I_shannon,
            "I_fused": I_fused,
        })

    df = pd.DataFrame(data)
    csv_path = os.path.join(primary, f"{prefix}fluctuation_samples.csv")
    df.to_csv(csv_path, index=False)
    _save_with_mirrors(csv_path, mirrors, put_in_figs=False)

    # ---------------------------
    # 6) Figures (mirror into <mirror>/figs/)
    # ---------------------------
    # E hist (linear)
    plt.figure()
    plt.hist(E0, bins=64)
    plt.xlabel("E0 (linear)")
    plt.ylabel("Count")
    plt.title("Energy distribution (linear)")
    f1 = os.path.join(figdir, f"{prefix}E_hist_linear.png")
    plt.savefig(f1, dpi=ACTIVE["RUNTIME"].get("matplotlib_dpi", 180), bbox_inches="tight")
    plt.close()
    _save_with_mirrors(f1, mirrors, put_in_figs=True)

    # E hist (log10)
    plt.figure()
    plt.hist(np.log10(E0 + 1e-12), bins=64)
    plt.xlabel("log10(E0)")
    plt.ylabel("Count")
    plt.title("Energy distribution (log10)")
    f2 = os.path.join(figdir, f"{prefix}E_hist_log.png")
    plt.savefig(f2, dpi=ACTIVE["RUNTIME"].get("matplotlib_dpi", 180), bbox_inches="tight")
    plt.close()
    _save_with_mirrors(f2, mirrors, put_in_figs=True)

    # EI-only plots
    f3 = f4 = None
    if use_I:
        # E vs I_fused scatter
        plt.figure()
        plt.scatter(E0, I_fused, s=6, alpha=0.5)
        plt.xlabel("E0")
        plt.ylabel("I_fused")
        plt.title("E vs I_fused")
        f3 = os.path.join(figdir, f"{prefix}E_vs_I_scatter.png")
        plt.savefig(f3, dpi=ACTIVE["RUNTIME"].get("matplotlib_dpi", 180), bbox_inches="tight")
        plt.close()
        _save_with_mirrors(f3, mirrors, put_in_figs=True)

        # X distribution
        plt.figure()
        plt.hist(X, bins=64)
        plt.xlabel("X")
        plt.ylabel("Count")
        plt.title("X distribution (from E and I)")
        f4 = os.path.join(figdir, f"{prefix}X_distribution.png")
        plt.savefig(f4, dpi=ACTIVE["RUNTIME"].get("matplotlib_dpi", 180), bbox_inches="tight")
        plt.close()
        _save_with_mirrors(f4, mirrors, put_in_figs=True)

    # ---------------------------
    # 7) JSON summary
    # ---------------------------
    summary = {
        "env": paths["env"],
        "run_id": paths["run_id"],
        "mode": "EI" if use_I else "E",
        "counts": {
            "num_universes": int(N),
            "in_goldilocks_E": int(in_goldilocks_E.sum()),
        },
        "stats": {
            "E0": {
                "min": float(np.min(E0)),
                "max": float(np.max(E0)),
                "mean": float(np.mean(E0)),
                "std": float(np.std(E0)),
            },
            "X": {
                "min": float(np.min(X)),
                "max": float(np.max(X)),
                "mean": float(np.mean(X)),
                "std": float(np.std(X)),
            },
        },
        "files": {
            "csv": os.path.relpath(csv_path, start=primary),
            "figs": [os.path.relpath(p, start=primary) for p in [f1, f2, f3, f4] if p],
        },
    }
    json_path = os.path.join(primary, f"{prefix}fluctuation_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    _save_with_mirrors(json_path, mirrors, put_in_figs=False)

    print(f"[FLUCT] mode={summary['mode']} → saved under:\n  {primary}")
    if mirrors:
        print(f"[FLUCT] mirrored to:\n  " + "\n  ".join(mirrors))

    return {
        "paths": paths,
        "summary": summary,
        "arrays": {
            "E0": E0,
            "logE0": logE0,
            "I_kl": I_kl if use_I else None,
            "I_shannon": I_shannon if use_I else None,
            "I_fused": I_fused if use_I else None,
            "X": X,
            "in_goldilocks_E": in_goldilocks_E,
        },
        "dataframe": df,
    }


if __name__ == "__main__":
    run_fluctuation(ACTIVE)
