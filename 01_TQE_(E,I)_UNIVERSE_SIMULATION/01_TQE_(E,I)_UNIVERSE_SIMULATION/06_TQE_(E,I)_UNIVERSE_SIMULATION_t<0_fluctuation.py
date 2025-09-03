# fluctuation.py
# ===================================================================================
# Fluctuation stage for the TQE universe simulation.
# Samples Energy (E) for Monte Carlo universes, optionally computes Information (I)
# via KL divergence to uniform and Shannon entropy over a random quantum-like
# probability vector (superposition proxy). Supports E-only and E×I modes.
#
# Outputs:
#   - CSV with all sampled fields
#   - PNG figures (E histograms; plus E–I scatter and X distribution for EI mode)
#   - JSON summary (basic stats)
#
# Routing:
#   Uses io_paths.resolve_output_paths(ACTIVE) to decide primary folder (Colab Drive
#   if in Colab, otherwise Desktop), and mirrors if configured.
# ===================================================================================
# Author: Stefan Len
# ===================================================================================

from config import ACTIVE                      # active config resolved from profiles
from imports import *                          # common libs (np, pd, plt, scipy, etc.)
from io_paths import resolve_output_paths, ensure_colab_drive_mounted

import json
import shutil
from typing import Dict, Tuple

# ---------------------------
# Small helpers
# ---------------------------

def _rng(seed):
    """Create a reproducible RNG; if seed is None, derive one."""
    if seed is None:
        ss = np.random.SeedSequence()
        return np.random.default_rng(ss)
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
    # Complex gaussian amplitudes -> squared magnitudes -> normalize to 1
    a = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    p = np.abs(a) ** 2
    p /= p.sum()
    return p

def _info_components(p: np.ndarray, kl_eps: float) -> Tuple[float, float, float]:
    """
    Compute KL to uniform (normalized), Shannon entropy (normalized),
    and a fused I in [0,1] (fusion applied by caller).
    Returns (I_kl_norm, H_norm, one_minus_H_norm)
    """
    dim = p.size
    u = np.full(dim, 1.0 / dim)
    # KL(p||u) = sum p * log(p/u) ; normalize by log(dim) to get [0, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        kl = np.sum(p * (np.log(p + kl_eps) - np.log(u + kl_eps)))
    kl_norm = float(kl / np.log(dim)) if dim > 1 else 0.0
    kl_norm = float(np.clip(kl_norm, 0.0, 1.0))

    # Shannon entropy normalized: H_norm = H / log(dim) ∈ [0,1]
    H = entropy(p, base=np.e)
    H_norm = float(H / np.log(dim)) if dim > 1 else 0.0
    H_norm = float(np.clip(H_norm, 0.0, 1.0))

    # For an "information-content-like" score from Shannon, use (1 - H_norm)
    one_minus_H_norm = 1.0 - H_norm
    return kl_norm, H_norm, one_minus_H_norm

def _fuse_I(kl_norm: float, shannon_info: float, info_cfg: dict) -> float:
    """Fuse KL- and Shannon-based components into a single I ∈ [0,1]."""
    mode = info_cfg.get("fusion", "product")
    if mode == "product":
        val = kl_norm * shannon_info
    elif mode == "weighted":
        w_kl = float(info_cfg.get("weight_kl", 0.5))
        w_sh = float(info_cfg.get("weight_shannon", 0.5))
        s = (w_kl + w_sh) or 1.0
        w_kl /= s
        w_sh /= s
        val = w_kl * kl_norm + w_sh * shannon_info
    else:
        val = kl_norm * shannon_info

    # post-processing
    exp_ = float(info_cfg.get("exponent", 1.0))
    floor = float(info_cfg.get("floor_eps", 0.0))
    val = max(val, floor)
    if exp_ != 1.0:
        val = val ** exp_
    return float(np.clip(val, 0.0, 1.0))

def _couple_X(E: np.ndarray, I: np.ndarray, x_cfg: dict) -> np.ndarray:
    """Compute X = f(E, I) according to COUPLING_X settings."""
    mode   = x_cfg.get("mode", "product")
    alphaI = float(x_cfg.get("alpha_I", 0.8))
    powI   = float(x_cfg.get("I_power", 1.0))
    scale  = float(x_cfg.get("scale", 1.0))

    if I is None:
        # E-only mode: define X ≡ E (so downstream code can still run)
        X = E.astype(float)
    else:
        if mode == "E_plus_I":
            X = E + alphaI * I
        elif mode == "E_times_I_pow":
            X = E * (alphaI * I) ** powI
        else:
            # default "product"
            X = E * (alphaI * I)
    return scale * X

def _save_with_mirrors(src_path: str, mirrors: list):
    """Copy a freshly written file to mirror directories (preserving filename)."""
    for m in mirrors:
        try:
            dst = os.path.join(m, os.path.basename(src_path))
            shutil.copy2(src_path, dst)
        except Exception as e:
            print(f"[WARN] Mirror copy failed → {m}: {e}")

# ---------------------------
# Main stage
# ---------------------------

def run_fluctuation(active_cfg: Dict = ACTIVE) -> Dict:
    """
    Run the fluctuation stage.
    - Samples E from lognormal with optional truncation
    - If PIPELINE.use_information=True, also computes I and coupled X
    - Saves CSV/JSON/PNGs to the resolved output folders
    Returns a dict with basic handles and in-memory arrays.
    """
    # Resolve I/O paths and mount Drive if needed
    paths = resolve_output_paths(active_cfg)
    ensure_colab_drive_mounted(active_cfg)
    primary = paths["primary_run_dir"]
    figdir  = paths["fig_dir"]
    mirrors = paths["mirrors"]
    allow   = set(active_cfg["OUTPUTS"]["local"]["allow_exts"])

    # Prefix for filenames (E__ vs EI__)
    use_I = bool(active_cfg["PIPELINE"].get("use_information", True))
    prefix = "EI__" if use_I else "E__"

    # RNG
    seed = active_cfg["ENERGY"].get("seed")
    rng  = _rng(seed)

    # ---------------------------
    # 1) Sample Energy (E)
    # ---------------------------
    N       = int(active_cfg["ENERGY"]["num_universes"])
    mu      = float(active_cfg["ENERGY"]["log_mu"])
    sigma   = float(active_cfg["ENERGY"]["log_sigma"])
    t_low   = active_cfg["ENERGY"].get("trunc_low", None)
    t_high  = active_cfg["ENERGY"].get("trunc_high", None)

    # lognormal: draw log(E) ~ N(mu, sigma^2) → E = exp(logE)
    logE0 = rng.normal(loc=mu, scale=sigma, size=N).astype(float)
    E0    = np.exp(logE0)
    E0    = _apply_truncation(E0, t_low, t_high)

    # ---------------------------
    # 2) Optional Information (I) over random prob. vectors
    # ---------------------------
    I_kl = I_shannon = I_fused = None
    p_store = None  # optional: store a few sample probability vectors

    if use_I:
        info_cfg = active_cfg["INFORMATION"]
        dim      = int(info_cfg["hilbert_dim"])
        eps      = float(info_cfg["kl_eps"])

        # Prepare arrays
        I_kl      = np.zeros(N, dtype=float)
        H_norm    = np.zeros(N, dtype=float)
        I_shannon = np.zeros(N, dtype=float)
        I_fused   = np.zeros(N, dtype=float)

        # Generate per-universe probability vectors and compute I components
        for i in range(N):
            p = _random_prob_vec(dim, rng)
            kl_norm, h_norm, sh_info = _info_components(p, eps)
            I_kl[i]      = kl_norm
            H_norm[i]    = h_norm
            I_shannon[i] = sh_info
            I_fused[i]   = _fuse_I(kl_norm, sh_info, info_cfg)

    # ---------------------------
    # 3) Coupling X = f(E, I) (or X=E in E-only mode)
    # ---------------------------
    X = _couple_X(E0, I_fused if use_I else None, active_cfg["COUPLING_X"])

    # ---------------------------
    # 4) Goldilocks (heuristic flag on E; X-diagnostics are plotted)
    # ---------------------------
    gcfg = active_cfg["GOLDILOCKS"]
    if gcfg.get("mode", "dynamic") == "heuristic":
        c = float(gcfg.get("E_center", 4.0))
        w = float(gcfg.get("E_width",  4.0))
    else:
        # still provide a coarse heuristic window for early diagnostics
        c, w = float(mu + sigma), float(2.0 * sigma)
    in_goldilocks_E = (np.abs(E0 - c) <= (w / 2.0)).astype(int)

    # ---------------------------
    # 5) Build DataFrame & save CSV
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
    _save_with_mirrors(csv_path, mirrors)

    # ---------------------------
    # 6) Figures
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
    _save_with_mirrors(f1, mirrors)

    # E hist (log10)
    plt.figure()
    plt.hist(np.log10(E0 + 1e-12), bins=64)
    plt.xlabel("log10(E0)")
    plt.ylabel("Count")
    plt.title("Energy distribution (log10)")
    f2 = os.path.join(figdir, f"{prefix}E_hist_log.png")
    plt.savefig(f2, dpi=ACTIVE["RUNTIME"].get("matplotlib_dpi", 180), bbox_inches="tight")
    plt.close()
    _save_with_mirrors(f2, mirrors)

    # EI-only plots
    f3 = f4 = None
    if use_I:
        # E vs I_fused scatter
        plt.figure()
        plt.scatter(E0, I_fused, s=6, alpha=0.5)
        plt.xlabel("E0")
        plt.ylabel("I_fused")
        plt.title("E vs I_fused")
        f3 = os.path.join(figdir, "EI__E_vs_I_scatter.png")
        plt.savefig(f3, dpi=ACTIVE["RUNTIME"].get("matplotlib_dpi", 180), bbox_inches="tight")
        plt.close()
        _save_with_mirrors(f3, mirrors)

        # X distribution
        plt.figure()
        plt.hist(X, bins=64)
        plt.xlabel("X")
        plt.ylabel("Count")
        plt.title("X distribution (from E and I)")
        f4 = os.path.join(figdir, "EI__X_distribution.png")
        plt.savefig(f4, dpi=ACTIVE["RUNTIME"].get("matplotlib_dpi", 180), bbox_inches="tight")
        plt.close()
        _save_with_mirrors(f4, mirrors)

    # ---------------------------
    # 7) JSON summary
    # ---------------------------
    summary = {
        "env": paths["env"],
        "run_id": paths["run_id"],
        "mode": "EI" if use_I else "E",
        "counts": {
            "num_universes": int(N),
            "in_goldilocks_E": int(int(in_goldilocks_E.sum())),
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
            "figs": [p for p in [f1, f2, f3, f4] if p],
        },
    }
    json_path = os.path.join(primary, f"{prefix}fluctuation_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    _save_with_mirrors(json_path, mirrors)

    print(f"[FLUCT] mode={summary['mode']} → saved CSV/JSON/PNGs under:\n  {primary}")
    if mirrors:
        print(f"[FLUCT] mirrored to:\n  " + "\n  ".join(mirrors))

    return {
        "paths": paths,
        "summary": summary,
        "arrays": {
            "E0": E0,
            "logE0": logE0,
            "I_kl": I_kl,
            "I_shannon": I_shannon,
            "I_fused": I_fused,
            "X": X,
            "in_goldilocks_E": in_goldilocks_E,
        },
        "dataframe": df,
    }

# Optional: allow running this stage directly
if __name__ == "__main__":
    run_fluctuation(ACTIVE)
