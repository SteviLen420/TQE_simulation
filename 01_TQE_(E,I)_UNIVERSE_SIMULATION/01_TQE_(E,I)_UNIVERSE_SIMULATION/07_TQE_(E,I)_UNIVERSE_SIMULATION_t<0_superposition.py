# superposition.py
# ===================================================================================
# Superposition (Information channel) for TQE universe simulation
# - Computes I_shannon (from eigenvalue entropy) and I_kl (KL to uniform of diag)
# - Fuses to scalar I according to cfg (product/weighted + exponent + floor)
# - Couples with Energy E to form X via COUPLING_X (product / E_plus_I / E_times_I_pow)
# - Saves CSV with per-universe metrics, JSON summary, and PNG plots (hist, scatters)
#
# Author: Stefan Len
# ===================================================================================

from config import ACTIVE
from io_paths import resolve_output_paths, ensure_colab_drive_mounted
# NOTE: If you keep a centralized imports.py, you may use it — otherwise we import locally.
import os, json, time, pathlib, math
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional quantum deps
try:
    import qutip as qt
except Exception:
    qt = None

from shutil import copy2

def _mirror_file(src: pathlib.Path, mirrors: list, put_in_figs: bool = False, cfg: dict = ACTIVE):
    fig_sub = cfg["OUTPUTS"]["local"].get("fig_subdir", "figs")
    for m in mirrors:
        m = pathlib.Path(m)
        if put_in_figs:
            dst_dir = m / fig_sub
            dst_dir.mkdir(parents=True, exist_ok=True)
            copy2(src, dst_dir / src.name)
        else:
            copy2(src, m / src.name)

# ---------------------------
# Helpers
# ---------------------------
def _rng(seed: Optional[int] = None):
    """Create a fresh Generator (reproducible if seed given)."""
    return np.random.default_rng(seed if seed is not None else np.random.SeedSequence().generate_state(1)[0])

def _random_pure_state(d: int, rng: np.random.Generator) -> np.ndarray:
    """Haar-like random ket via complex normal + normalization."""
    psi = rng.normal(size=d) + 1j * rng.normal(size=d)
    psi = psi / np.linalg.norm(psi)
    return psi

def _depolarize_rho(rho: np.ndarray, lam: float) -> np.ndarray:
    """ρ' = (1-λ)ρ + λ I/d  (keeps PSD and trace=1)."""
    d = rho.shape[0]
    return (1.0 - lam) * rho + (lam / d) * np.eye(d, dtype=complex)

def _von_neumann_entropy_eigs(eigs: np.ndarray, base: float = 2.0) -> float:
    """S(ρ) = -Tr(ρ log ρ). Input: eigenvalues (>=0, sum=1)."""
    eps = 1e-15
    x = np.clip(eigs.real, eps, 1.0)
    return float(-(x * (np.log(x) / np.log(base))).sum())

def _kl_to_uniform(prob: np.ndarray, base: float = 2.0) -> float:
    """D_KL(p || u) where u is uniform on d outcomes. Returns in 'base' units, e.g. bits."""
    d = prob.size
    u = 1.0 / d
    eps = 1e-15
    p = np.clip(prob.real, eps, 1.0)
    return float((p * ((np.log(p) - np.log(u)) / np.log(base))).sum())

def _fuse_I(I_kl: float, I_sh: float, cfg: dict) -> float:
    """Fuse information components to scalar I in [0,1] per config."""
    mode = cfg["INFORMATION"]["fusion"]
    if mode == "product":
        I = I_kl * I_sh
    elif mode == "weighted":
        w_kl = cfg["INFORMATION"].get("weight_kl", 0.5)
        w_sh = cfg["INFORMATION"].get("weight_shannon", 0.5)
        s = max(w_kl + w_sh, 1e-12)
        I = (w_kl * I_kl + w_sh * I_sh) / s
    else:
        I = 0.5 * (I_kl + I_sh)

    # post-processing
    expn = cfg["INFORMATION"].get("exponent", 1.0)
    floor = cfg["INFORMATION"].get("floor_eps", 0.0)
    I = max(I, floor)
    if expn != 1.0:
        I = I ** expn
    # clamp to [0,1]
    return float(np.clip(I, 0.0, 1.0))

def _couple_X(E: float, I: Optional[float], cfg: dict) -> float:
    """X = f(E,I) according to COUPLING_X; if I is None (E-only), degrade to pure E."""
    mode   = cfg["COUPLING_X"]["mode"]
    alpha  = cfg["COUPLING_X"]["alpha_I"]
    scale  = cfg["COUPLING_X"]["scale"]
    if I is None:
        return float(scale * E)

    if mode == "product":
        X = E * (alpha * I)
    elif mode == "E_plus_I":
        X = E + alpha * I
    elif mode == "E_times_I_pow":
        p = cfg["COUPLING_X"].get("I_power", 1.0)
        X = E * ((alpha * I) ** p)
    else:
        X = E * (alpha * I)
    return float(scale * X)

# ---------------------------
# Core: compute information (I) for N universes
# ---------------------------
def _compute_information_for_population(N: int, cfg: dict, seed: Optional[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns arrays (I_shannon, I_kl, I_fused) of length N in [0,1].
    Strategy:
      - Build (mixed) states: ρ' = (1-λ)|ψ⟩⟨ψ| + λ I/d, with λ ~ U[0, λ_max]
      - I_shannon: 1 - S(ρ') / log_d          (global mixedness; von Neumann entropy)
      - I_kl:      KL(diag(ρ') || uniform)/log_d  (basis-population surprise vs uniform)
    These two differ because I_kl discards coherence (basis diag), I_shannon uses spectrum.
    """
    d        = cfg["INFORMATION"]["hilbert_dim"]
    kl_eps   = cfg["INFORMATION"]["kl_eps"]
    rng      = _rng(seed)
    lam_max  = 0.35  # mild depolarization range to generate variety

    I_sh_list, I_kl_list, I_list = [], [], []

    use_qutip = (qt is not None)

    for _ in range(N):
        # random pure state
        psi = _random_pure_state(d, rng)

        if use_qutip:
            ket = qt.Qobj(psi.reshape((d, 1)))
            rho = ket * ket.dag()  # |ψ⟩⟨ψ|
            # depolarize
            lam = rng.uniform(0.0, lam_max)
            rho = (1.0 - lam) * rho + lam * (qt.qeye(d) / d)
            # eigenvalues for von Neumann entropy
            eigs = np.sort(np.maximum(rho.eigenenergies().real, 0.0))[::-1]
            eigs = eigs / eigs.sum()
            S = _von_neumann_entropy_eigs(eigs, base=2.0)  # bits
            # diagonal probabilities in computational basis
            diag = np.real(np.clip(rho.diag().full().ravel(), 0.0, 1.0))
        else:
            # NumPy proxy
            rho = np.outer(psi, np.conjugate(psi))
            lam = rng.uniform(0.0, lam_max)
            rho = _depolarize_rho(rho, lam)
            # eigenvalues
            eigs = np.linalg.eigvalsh(rho)
            eigs = np.clip(eigs.real, 0.0, 1.0)
            s = eigs.sum()
            eigs = eigs / (s if s > 0 else 1.0)
            S = _von_neumann_entropy_eigs(eigs, base=2.0)
            # diagonal (computational basis)
            diag = np.clip(np.real(np.diag(rho)), 0.0, 1.0)

        # normalize by log d
        logd = math.log(d, 2)
        I_sh = float(np.clip(1.0 - S / logd, 0.0, 1.0))

        # KL vs uniform (basis-population)
        Dkl = _kl_to_uniform(diag, base=2.0)
        I_kl = float(np.clip(Dkl / (logd + kl_eps), 0.0, 1.0))

        I = _fuse_I(I_kl, I_sh, cfg)

        I_sh_list.append(I_sh)
        I_kl_list.append(I_kl)
        I_list.append(I)

    return np.array(I_sh_list), np.array(I_kl_list), np.array(I_list)

# ---------------------------
# Public API
# ---------------------------
def run_superposition(E: Optional[np.ndarray] = None, cfg: dict = ACTIVE, seed: Optional[int] = None) -> Dict[str, str]:
    """
    Compute superposition/information metrics for a population and couple to energy E.
    If E is None, this function does not sample energy; it only computes I arrays.
    Returns dict with paths (primary_run_dir, fig_dir) to integrate with later blocks.

    CSV columns: ['universe_id', 'E', 'I_shannon', 'I_kl', 'I', 'X']
    JSON summary: means/stds + environment info.
    PNG: histogram of I, scatter E-vs-I, scatter E-vs-X (if E is given).
    """
    # Resolve output destinations
    ensure_colab_drive_mounted(cfg)
    paths = resolve_output_paths(cfg)
    run_dir = pathlib.Path(paths["primary_run_dir"])
    fig_dir = pathlib.Path(paths["fig_dir"])

    # Determine population size N
    if E is None:
        N = cfg["ENERGY"]["num_universes"]
    else:
        N = int(len(E))

    # If information channel is disabled, short-circuit (E-only)
    use_info = cfg["PIPELINE"].get("use_information", True)
    if not use_info:
        # Create a minimal CSV with NaN info and X=E (pure-E coupling)
        df = pd.DataFrame({
            "universe_id": np.arange(N, dtype=int),
            "E": np.asarray(E) if E is not None else np.nan,
            "I_shannon": np.nan,
            "I_kl": np.nan,
            "I": np.nan,
            "X": np.asarray(E) if E is not None else np.nan,
        })
        csv_path = run_dir / "superposition_E_only.csv"
        df.to_csv(csv_path, index=False)

        # Summary JSON
        summary = {
            "env": paths["env"],
            "run_id": paths["run_id"],
            "N": N,
            "mode": "E-only",
            "has_energy_input": E is not None,
        }
        with open(run_dir / "superposition_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        # No plots (no I available); still return directories
        return {"primary_run_dir": str(run_dir), "fig_dir": str(fig_dir)}

    # Otherwise: compute information channel
    I_sh, I_kl, I = _compute_information_for_population(N, cfg, seed)

    # Couple with energy if provided
    if E is None:
        E_arr = np.full(N, np.nan)
        X_arr = np.full(N, np.nan)
    else:
        E_arr = np.asarray(E, dtype=float)
        X_arr = np.array([_couple_X(float(e), float(i), cfg) for e, i in zip(E_arr, I)])

    # Assemble DataFrame
    df = pd.DataFrame({
        "universe_id": np.arange(N, dtype=int),
        "E": E_arr,
        "I_shannon": I_sh,
        "I_kl": I_kl,
        "I": I,
        "X": X_arr,
    })
    csv_path = run_dir / "superposition.csv"
    df.to_csv(csv_path, index=False)

    # Summary JSON
    def _safe_stats(x):
        x = x[np.isfinite(x)]
        return {"mean": float(np.mean(x)) if x.size else None,
                "std":  float(np.std(x))  if x.size else None,
                "min":  float(np.min(x))  if x.size else None,
                "max":  float(np.max(x))  if x.size else None,
                "n":    int(x.size)}

    summary = {
        "env": paths["env"],
        "run_id": paths["run_id"],
        "N": N,
        "mode": "E×I",
        "I_shannon": _safe_stats(I_sh),
        "I_kl": _safe_stats(I_kl),
        "I": _safe_stats(I),
        "E": _safe_stats(E_arr) if np.isfinite(E_arr).any() else None,
        "X": _safe_stats(X_arr) if np.isfinite(X_arr).any() else None,
        "cfg": {
            "hilbert_dim": cfg["INFORMATION"]["hilbert_dim"],
            "fusion": cfg["INFORMATION"]["fusion"],
            "alpha_I": cfg["COUPLING_X"]["alpha_I"],
            "mode": cfg["COUPLING_X"]["mode"],
        }
    }
    with open(run_dir / "superposition_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ---------------------------
    # Plots
    # ---------------------------
    # 1) Histogram of fused I
    plt.figure()
    plt.hist(I, bins=40)
    plt.xlabel("I (fused)")
    plt.ylabel("count")
    plt.title("Superposition — I distribution")
    plt.tight_layout()
    fig_path1 = fig_dir / "superposition_I_hist.png"
    plt.savefig(fig_path1, dpi=cfg["RUNTIME"].get("matplotlib_dpi", 180))
    plt.close()

    # 2) Scatter E vs I (if E provided)
    if np.isfinite(E_arr).any():
        plt.figure()
        plt.scatter(E_arr, I, s=6, alpha=0.6)
        plt.xlabel("E (energy)")
        plt.ylabel("I (fused)")
        plt.title("E vs I")
        plt.tight_layout()
        fig_path2 = fig_dir / "superposition_E_vs_I.png"
        plt.savefig(fig_path2, dpi=cfg["RUNTIME"].get("matplotlib_dpi", 180))
        plt.close()

        # 3) Scatter E vs X
        if np.isfinite(X_arr).any():
            plt.figure()
            plt.scatter(E_arr, X_arr, s=6, alpha=0.6)
            plt.xlabel("E (energy)")
            plt.ylabel("X = f(E,I)")
            plt.title("E vs X")
            plt.tight_layout()
            fig_path3 = fig_dir / "superposition_E_vs_X.png"
            plt.savefig(fig_path3, dpi=cfg["RUNTIME"].get("matplotlib_dpi", 180))
            plt.close()

    # Done
    return {"primary_run_dir": str(run_dir), "fig_dir": str(fig_dir)}
