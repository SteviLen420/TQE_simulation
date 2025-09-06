# ===================================================================================
# TQE_08_EI_UNIVERSE_SIMULATION_t_lt_0_superposition.py
# ===================================================================================
# Author: Stefan Len
# ===================================================================================

from typing import Optional, Tuple, Dict, List
import os, json, math, pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shutil import copy2

# Cached config + resolved paths for the current run (stable run_id)
from TQE_03_EI_UNIVERSE_SIMULATION_imports import ACTIVE, PATHS, RUN_DIR, FIG_DIR
from TQE_04_EI_UNIVERSE_SIMULATION_seeding import load_or_create_run_seeds

# Optional quantum dependency (graceful fallback to NumPy)
try:
    import qutip as qt
except Exception:
    qt = None

# -----------------------------------------------------------------------------------
# Small I/O helper: mirror a freshly written file to all mirror directories.
# If put_in_figs=True, file will be placed inside each mirror's <fig_subdir>/.
# -----------------------------------------------------------------------------------
def _mirror_file(src: pathlib.Path, mirrors: List[str], put_in_figs: bool, cfg: dict) -> None:
    fig_sub = cfg["OUTPUTS"]["local"].get("fig_subdir", "figs")
    for m in mirrors or []:
        try:
            mpath = pathlib.Path(m)
            if put_in_figs:
                (mpath / fig_sub).mkdir(parents=True, exist_ok=True)
                copy2(src, mpath / fig_sub / src.name)
            else:
                mpath.mkdir(parents=True, exist_ok=True)
                copy2(src, mpath / src.name)
        except Exception as e:
            print(f"[WARN] Mirror copy failed → {m}: {e}")

# -----------------------------------------------------------------------------------
# RNG and quantum helpers
# -----------------------------------------------------------------------------------

def _random_pure_state(d: int, rng: np.random.Generator) -> np.ndarray:
    """Haar-like random ket via complex normal + normalization."""
    psi = rng.normal(size=d) + 1j * rng.normal(size=d)
    return psi / np.linalg.norm(psi)

def _depolarize_rho(rho: np.ndarray, lam: float) -> np.ndarray:
    """ρ' = (1-λ)ρ + λ I/d  (keeps PSD and trace=1)."""
    d = rho.shape[0]
    return (1.0 - lam) * rho + (lam / d) * np.eye(d, dtype=complex)

def _von_neumann_entropy_eigs(eigs: np.ndarray, base: float = 2.0) -> float:
    """S(ρ) = -Σ λ log_base λ . Input: eigenvalues (>=0, sum=1)."""
    eps = 1e-15
    x = np.clip(eigs.real, eps, 1.0)
    return float(-(x * (np.log(x) / np.log(base))).sum())

def _kl_to_uniform(prob: np.ndarray, base: float = 2.0) -> float:
    """D_KL(p || u) where u is uniform on d outcomes. Returns in 'base' units (bits)."""
    d = prob.size
    u = 1.0 / d
    eps = 1e-15
    p = np.clip(prob.real, eps, 1.0)
    return float((p * ((np.log(p) - np.log(u)) / np.log(base))).sum())

def _fuse_I(I_kl: float, I_sh: float, cfg: dict) -> float:
    """Fuse information components to scalar I in [0,1] per config."""
    info_cfg = cfg["INFORMATION"]
    mode = info_cfg.get("fusion", "product")

    if mode == "product":
        I = I_kl * I_sh
    elif mode == "weighted":
        w_kl = float(info_cfg.get("weight_kl", 0.5))
        w_sh = float(info_cfg.get("weight_shannon", 0.5))
        s = max(w_kl + w_sh, 1e-12)
        I = (w_kl * I_kl + w_sh * I_sh) / s
    else:
        I = 0.5 * (I_kl + I_sh)

    # Post-processing
    expn  = float(info_cfg.get("exponent", 1.0))
    floor = float(info_cfg.get("floor_eps", 0.0))
    I = max(I, floor)
    if expn != 1.0:
        I = I ** expn
    return float(np.clip(I, 0.0, 1.0))

def _couple_X(E: float, I: Optional[float], cfg: dict) -> float:
    """X = f(E,I) according to COUPLING_X; if I is None, degrade to pure E."""
    xcfg  = cfg["COUPLING_X"]
    mode  = xcfg.get("mode", "product")
    alpha = float(xcfg.get("alpha_I", 0.8))
    scale = float(xcfg.get("scale", 1.0))

    if I is None or not np.isfinite(I):
        return float(scale * E)

    if mode == "E_plus_I":
        X = E + alpha * I
    elif mode == "E_times_I_pow":
        p = float(xcfg.get("I_power", 1.0))
        X = E * ((alpha * I) ** p)
    else:  # "product" default
        X = E * (alpha * I)
    return float(scale * X)

# -----------------------------------------------------------------------------------
# Core: compute information (I) arrays for N universes (NumPy or QuTiP path)
# -----------------------------------------------------------------------------------
def _compute_information_for_population(N: int, cfg: dict, seed: Optional[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns arrays (I_shannon, I_kl, I_fused) of length N in [0,1].
      - Build (mixed) states: ρ' = (1-λ)|ψ⟩⟨ψ| + λ I/d with λ ~ U[0, λ_max]
      - I_shannon: 1 - S(ρ') / log_d            (spectral mixedness)
      - I_kl:      KL(diag(ρ') || uniform)/log_d (basis-population surprise)
    """
    d       = int(cfg["INFORMATION"]["hilbert_dim"])
    kl_eps  = float(cfg["INFORMATION"]["kl_eps"])
    # RNG from central seeder
    seeds_data = load_or_create_run_seeds(ACTIVE)
    master_seed = seeds_data["master_seed"]
    rng = np.random.default_rng(master_seed)
    lam_max = 0.35  # small depolarization for diversity

    I_sh_list, I_kl_list, I_list = [], [], []
    use_qutip = (qt is not None)

    for _ in range(N):
        # random pure state
        psi = _random_pure_state(d, rng)

        if use_qutip:
            # Build density matrix in QuTiP
            ket = qt.Qobj(psi.reshape((d, 1)))
            rho = ket * ket.dag()  # |ψ><ψ|
            lam = rng.uniform(0.0, lam_max)
            rho = (1.0 - lam) * rho + lam * (qt.qeye(d) / d)

            # Eigenvalues (sorted), diagonal as measurement probs in computational basis
            eigs = np.clip(np.real(np.sort(rho.eigenenergies())[::-1]), 0.0, 1.0)
            eigs /= eigs.sum() if eigs.sum() > 0 else 1.0
            S = _von_neumann_entropy_eigs(eigs, base=2.0)
            diag = np.clip(np.array(rho.diag()).astype(float).ravel(), 0.0, 1.0)
            diag /= diag.sum() if diag.sum() > 0 else 1.0
        else:
            # NumPy path
            rho = np.outer(psi, np.conjugate(psi))
            lam = rng.uniform(0.0, lam_max)
            rho = _depolarize_rho(rho, lam)

            eigs = np.clip(np.linalg.eigvalsh(rho).real, 0.0, 1.0)
            eigs /= eigs.sum() if eigs.sum() > 0 else 1.0
            S = _von_neumann_entropy_eigs(eigs, base=2.0)

            diag = np.clip(np.diag(rho).real, 0.0, 1.0)
            diag /= diag.sum() if diag.sum() > 0 else 1.0

        logd = math.log(d, 2)
        I_sh  = float(np.clip(1.0 - S / logd, 0.0, 1.0))
        Dkl   = _kl_to_uniform(diag, base=2.0)
        Ikl_n = float(np.clip(Dkl / (logd + kl_eps), 0.0, 1.0))
        Ifuse = _fuse_I(Ikl_n, I_sh, cfg)

        I_sh_list.append(I_sh)
        I_kl_list.append(Ikl_n)
        I_list.append(Ifuse)

    return np.asarray(I_sh_list), np.asarray(I_kl_list), np.asarray(I_list)

# -----------------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------------
def run_superposition(E: Optional[np.ndarray] = None, cfg: dict = ACTIVE, seed: Optional[int] = None) -> Dict[str, str]:
    """
    Compute superposition/information metrics and couple to energy E if provided.
    Uses the cached RUN_DIR/FIG_DIR/PATHS for consistent run IDs across the pipeline.
    """
    # Early exit if the whole stage is disabled
    if not cfg["PIPELINE"].get("run_superposition", True):
        return {}

    # Resolved (cached) destinations
    paths   = PATHS
    run_dir = pathlib.Path(RUN_DIR)
    fig_dir = pathlib.Path(FIG_DIR)
    mirrors = paths.get("mirrors", [])

    # Filename tag prefix (EI__/E__) if requested
    ei_tag_enabled = cfg["OUTPUTS"].get("tag_ei_in_filenames", True)
    use_info       = cfg["PIPELINE"].get("use_information", True)
    tag_prefix     = ("EI__" if use_info else "E__") if ei_tag_enabled else ""

    # Stage-level save switches
    per_stage = cfg["OUTPUTS"].get("save_per_stage", {})
    save_stage = bool(per_stage.get("superposition", True))
    save_csv   = save_stage and bool(cfg["OUTPUTS"].get("save_csv", True))
    save_figs  = save_stage and bool(cfg["OUTPUTS"].get("save_figs", True))
    save_json  = save_stage and bool(cfg["OUTPUTS"].get("save_json", True))

    # Population size and seed
    N = int(cfg["ENERGY"]["num_universes"] if E is None else len(E))
    seed_eff = cfg["ENERGY"].get("seed") if seed is None else seed

    # If information channel is OFF → E-only short-circuit
    if not use_info:
        E_arr = np.asarray(E) if E is not None else np.full(N, np.nan)
        X_arr = E_arr.copy()
        df = pd.DataFrame({
            "universe_id": np.arange(N, dtype=int),
            "E": E_arr,
            "I_shannon": np.nan,
            "I_kl": np.nan,
            "I": np.nan,
            "X": X_arr,
        })
        if save_csv:
            csv_path = run_dir / f"{tag_prefix}superposition_E_only.csv"
            df.to_csv(csv_path, index=False)
            _mirror_file(csv_path, mirrors, put_in_figs=False, cfg=cfg)
        if save_json:
            summary = {
                "env": paths["env"],
                "run_id": paths["run_id"],
                "N": N,
                "mode": "E-only",
                "has_energy_input": E is not None,
            }
            jpath = run_dir / f"{tag_prefix}superposition_summary.json"
            with open(jpath, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            _mirror_file(jpath, mirrors, put_in_figs=False, cfg=cfg)
        return {"primary_run_dir": str(run_dir), "fig_dir": str(fig_dir)}

    # Compute information channel
    I_sh, I_kl, I = _compute_information_for_population(N, cfg, seed_eff)

    # Couple with energy if provided
    if E is None:
        E_arr = np.full(N, np.nan)
        X_arr = np.full(N, np.nan)
    else:
        E_arr = np.asarray(E, dtype=float)
        X_arr = np.array([_couple_X(float(e), float(i), cfg) for e, i in zip(E_arr, I)], dtype=float)

    # Build DataFrame
    df = pd.DataFrame({
        "universe_id": np.arange(N, dtype=int),
        "E": E_arr,
        "I_shannon": I_sh,
        "I_kl": I_kl,
        "I": I,
        "X": X_arr,
    })

    # Save CSV
    if save_csv:
        csv_path = run_dir / f"{tag_prefix}superposition.csv"
        df.to_csv(csv_path, index=False)
        _mirror_file(csv_path, mirrors, put_in_figs=False, cfg=cfg)

    # Summary JSON
    if save_json:
        def _stats(x: np.ndarray):
            x = x[np.isfinite(x)]
            if x.size == 0:
                return {"n": 0, "min": None, "max": None, "mean": None, "std": None}
            return {
                "n": int(x.size),
                "min": float(np.min(x)),
                "max": float(np.max(x)),
                "mean": float(np.mean(x)),
                "std": float(np.std(x)),
            }

        summary = {
            "env": paths["env"],
            "run_id": paths["run_id"],
            "N": N,
            "mode": "E×I",
            "stats": {
                "I_shannon": _stats(I_sh),
                "I_kl": _stats(I_kl),
                "I": _stats(I),
                "E": _stats(E_arr) if np.isfinite(E_arr).any() else None,
                "X": _stats(X_arr) if np.isfinite(X_arr).any() else None,
            },
            "cfg": {
                "hilbert_dim": cfg["INFORMATION"]["hilbert_dim"],
                "fusion": cfg["INFORMATION"]["fusion"],
                "coupling_mode": cfg["COUPLING_X"]["mode"],
                "alpha_I": cfg["COUPLING_X"]["alpha_I"],
            }
        }
        jpath = run_dir / f"{tag_prefix}superposition_summary.json"
        with open(jpath, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        _mirror_file(jpath, mirrors, put_in_figs=False, cfg=cfg)

    # Plots
    if save_figs:
        dpi = int(cfg["RUNTIME"].get("matplotlib_dpi", 180))

        # 1) Histogram of fused I
        plt.figure()
        plt.hist(I, bins=40)
        plt.xlabel("I (fused)")
        plt.ylabel("Count")
        plt.title("Superposition — I distribution")
        plt.tight_layout()
        f1 = fig_dir / f"{tag_prefix}superposition_I_hist.png"
        plt.savefig(f1, dpi=dpi)
        plt.close()
        _mirror_file(f1, mirrors, put_in_figs=True, cfg=cfg)

        # 2) Scatter E vs I (if E provided)
        if np.isfinite(E_arr).any():
            plt.figure()
            plt.scatter(E_arr, I, s=6, alpha=0.6)
            plt.xlabel("E (energy)")
            plt.ylabel("I (fused)")
            plt.title("E vs I")
            plt.tight_layout()
            f2 = fig_dir / f"{tag_prefix}superposition_E_vs_I.png"
            plt.savefig(f2, dpi=dpi)
            plt.close()
            _mirror_file(f2, mirrors, put_in_figs=True, cfg=cfg)

            # 3) Scatter E vs X (if X is finite)
            if np.isfinite(X_arr).any():
                plt.figure()
                plt.scatter(E_arr, X_arr, s=6, alpha=0.6)
                plt.xlabel("E (energy)")
                plt.ylabel("X = f(E,I)")
                plt.title("E vs X")
                plt.tight_layout()
                f3 = fig_dir / f"{tag_prefix}superposition_E_vs_X.png"
                plt.savefig(f3, dpi=dpi)
                plt.close()
                _mirror_file(f3, mirrors, put_in_figs=True, cfg=cfg)

    return {"primary_run_dir": str(run_dir), "fig_dir": str(fig_dir)}

# Thin wrapper to match Master Control entrypoint
def run_superposition_stage(active: Dict = ACTIVE, seed: Optional[int] = None) -> Dict[str, str]:
    return run_superposition(E=None, cfg=active, seed=seed)

if __name__ == "__main__":
    run_superposition(cfg=ACTIVE)
