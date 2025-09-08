# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Stefan Len

# ===================================================================================
# TQE_08_EI_UNIVERSE_SIMULATION_t_lt_0_superposition.py
# ===================================================================================
# Author: Stefan Len
# ===================================================================================
#
# SUMMARY:
# This script simulates an optional quantum superposition stage (t < 0), providing a
# physically motivated and sophisticated model for the "Information" (I) component
# of the simulation. It replaces simpler probabilistic models with ones based on
# quantum mechanics, optionally using the QuTiP library.
#
# For each universe, it constructs a random quantum state (a density matrix ρ) by
# taking a pure state |ψ⟩ and mixing it with classical noise (depolarization).
# It then extracts two distinct information metrics from this quantum state:
# 1.  Shannon-based Information (I_sh): Derived from the state's Von Neumann entropy,
#     quantifying its degree of quantum mixedness or purity.
# 2.  KL-based Information (I_kl): Derived from the measurement probabilities
#     (the diagonal of ρ), quantifying the state's non-uniformity.
#
# These components are fused into a single value, I. The script can either
# generate these I values or take Energy (E) values as input to compute the
# final coupled variable X = f(E, I). It produces the standard CSV, PNG, and
# JSON outputs for the stage.
#
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

# Optional dependency
try:
    import qutip as qt
except Exception:
    qt = None


# -----------------------------------------------------------------------------------
# I/O helper
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
# RNG / quantum helpers
# -----------------------------------------------------------------------------------
def _random_pure_state(d: int, rng: np.random.Generator) -> np.ndarray:
    psi = rng.normal(size=d) + 1j * rng.normal(size=d)
    return psi / np.linalg.norm(psi)

def _depolarize_rho(rho: np.ndarray, lam: float) -> np.ndarray:
    d = rho.shape[0]
    return (1.0 - lam) * rho + (lam / d) * np.eye(d, dtype=complex)

def _von_neumann_entropy_eigs(eigs: np.ndarray, base: float = 2.0) -> float:
    eps = 1e-15
    x = np.clip(eigs.real, eps, 1.0)
    return float(-(x * (np.log(x) / np.log(base))).sum())

def _kl_to_uniform(prob: np.ndarray, base: float = 2.0) -> float:
    d = prob.size
    u = 1.0 / d
    eps = 1e-15
    p = np.clip(prob.real, eps, 1.0)
    return float((p * ((np.log(p) - np.log(u)) / np.log(base))).sum())

def _fuse_I(I_kl: float, I_sh: float, cfg: dict) -> float:
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

    expn  = float(info_cfg.get("exponent", 1.0))
    floor = float(info_cfg.get("floor_eps", 0.0))
    I = max(I, floor)
    if expn != 1.0:
        I = I ** expn
    return float(np.clip(I, 0.0, 1.0))

def _couple_X(E: float, I: Optional[float], cfg: dict) -> float:
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
    else:  # default "product"
        X = E * (alpha * I)
    return float(scale * X)


# -----------------------------------------------------------------------------------
# Compute information arrays
# -----------------------------------------------------------------------------------
def _compute_information_for_population(N: int, cfg: dict, seed: Optional[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (I_shannon, I_kl, I_fused) in [0,1], each length N.
    ρ' = (1-λ)|ψ⟩⟨ψ| + λ I/d, λ ~ U[0, λ_max]
    """
    d      = int(cfg["INFORMATION"]["hilbert_dim"])
    kl_eps = float(cfg["INFORMATION"]["kl_eps"])

    # Seed handling: use explicit seed if provided; else central seeder
    if seed is None:
        seeds_data = load_or_create_run_seeds(cfg)
        master_seed = seeds_data["master_seed"]
    else:
        master_seed = int(seed)
    rng = np.random.default_rng(master_seed)

    lam_max = 0.35
    I_sh_list, I_kl_list, I_list = [], [], []
    use_qutip = (qt is not None)

    for _ in range(N):
        psi = _random_pure_state(d, rng)

        if use_qutip:
            ket = qt.Qobj(psi.reshape((d, 1)))
            rho = ket * ket.dag()
            lam = rng.uniform(0.0, lam_max)
            rho = (1.0 - lam) * rho + lam * (qt.qeye(d) / d)

            eigs = np.clip(np.real(np.sort(rho.eigenenergies())[::-1]), 0.0, 1.0)
            eigs /= eigs.sum() if eigs.sum() > 0 else 1.0
            S = _von_neumann_entropy_eigs(eigs, base=2.0)

            diag = np.clip(np.array(rho.diag()).astype(float).ravel(), 0.0, 1.0)
            diag /= diag.sum() if diag.sum() > 0 else 1.0
        else:
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
# Worker + entrypoint
# -----------------------------------------------------------------------------------
def run_superposition(active_cfg: dict, arrays: Optional[dict] = None, seed: Optional[int] = None) -> Dict:
    """Compute I (and X if E provided), save artifacts, return dataframe + arrays."""
    if not active_cfg["PIPELINE"].get("run_superposition", True):
        return {}

    # Resolved (cached) destinations
    paths   = PATHS
    run_dir = pathlib.Path(RUN_DIR)
    fig_dir = pathlib.Path(FIG_DIR)
    mirrors = paths.get("mirrors", [])

    # Energy input from previous stage (optional)
    E = arrays.get("E0") if arrays else None

    # Filename tag prefix
    ei_tag_enabled = active_cfg["OUTPUTS"].get("tag_ei_in_filenames", True)
    use_info       = active_cfg["PIPELINE"].get("use_information", True)
    tag_prefix     = ("EI__" if use_info else "E__") if ei_tag_enabled else ""

    # Stage-level save switches
    per_stage = active_cfg["OUTPUTS"].get("save_per_stage", {})
    save_stage = bool(per_stage.get("superposition", True))
    save_csv   = save_stage and bool(active_cfg["OUTPUTS"].get("save_csv", True))
    save_figs  = save_stage and bool(active_cfg["OUTPUTS"].get("save_figs", True))
    save_json  = save_stage and bool(active_cfg["OUTPUTS"].get("save_json", True))

    # Population size
    N = int(active_cfg["ENERGY"]["num_universes"] if E is None else len(E))

    # E-only mode
    if not use_info:
        E_arr = np.asarray(E) if E is not None else np.full(N, np.nan)
        X_arr = E_arr.copy()
        df = pd.DataFrame({
            "universe_id": np.arange(N, dtype=int),
            "E": E_arr, "I_shannon": np.nan, "I_kl": np.nan, "I": np.nan, "X": X_arr
        })

        if save_csv:
            csv_path = run_dir / f"{tag_prefix}superposition_E_only.csv"
            df.to_csv(csv_path, index=False)
            _mirror_file(csv_path, mirrors, put_in_figs=False, cfg=active_cfg)

        if save_json:
            summary = {"run_id": paths["run_id"], "N": N, "mode": "E-only", "has_energy_input": E is not None}
            jpath = run_dir / f"{tag_prefix}superposition_summary.json"
            with open(jpath, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            _mirror_file(jpath, mirrors, put_in_figs=False, cfg=active_cfg)

        return {"dataframe": df, "arrays": {"E0": E_arr, "X": X_arr}}

    # E–I mode
    I_sh, I_kl, I = _compute_information_for_population(N, active_cfg, seed)

    if E is None:
        E_arr = np.full(N, np.nan)
        X_arr = np.full(N, np.nan)
    else:
        E_arr = np.asarray(E, dtype=float)
        X_arr = np.array([_couple_X(float(e), float(i), active_cfg) for e, i in zip(E_arr, I)], dtype=float)

    df = pd.DataFrame({
        "universe_id": np.arange(N, dtype=int),
        "E": E_arr, "I_shannon": I_sh, "I_kl": I_kl, "I": I, "X": X_arr
    })

    if save_csv:
        csv_path = run_dir / f"{tag_prefix}superposition.csv"
        df.to_csv(csv_path, index=False)
        _mirror_file(csv_path, mirrors, put_in_figs=False, cfg=active_cfg)

    if save_json:
        def _stats(x: np.ndarray):
            x = x[np.isfinite(x)]
            if x.size == 0:
                return {"n": 0, "min": None, "max": None, "mean": None, "std": None}
            return {"n": int(x.size), "min": float(np.min(x)), "max": float(np.max(x)),
                    "mean": float(np.mean(x)), "std": float(np.std(x))}
        summary = {
            "run_id": paths["run_id"], "N": N, "mode": "E×I",
            "stats": {"I_shannon": _stats(I_sh), "I_kl": _stats(I_kl), "I": _stats(I),
                      "E": _stats(E_arr) if np.isfinite(E_arr).any() else None,
                      "X": _stats(X_arr) if np.isfinite(X_arr).any() else None}
        }
        jpath = run_dir / f"{tag_prefix}superposition_summary.json"
        with open(jpath, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        _mirror_file(jpath, mirrors, put_in_figs=False, cfg=active_cfg)

    if save_figs:
        dpi = int(active_cfg["RUNTIME"].get("matplotlib_dpi", 180))
        plt.figure(); plt.hist(I, bins=40)
        plt.xlabel("I (fused)"); plt.ylabel("Count"); plt.title("Superposition — I distribution")
        f1 = fig_dir / f"{tag_prefix}superposition_I_hist.png"
        plt.savefig(f1, dpi=dpi, bbox_inches="tight"); plt.close()
        _mirror_file(f1, mirrors, put_in_figs=True, cfg=active_cfg)

        if np.isfinite(E_arr).any():
            plt.figure(); plt.scatter(E_arr, I, s=6, alpha=0.6)
            plt.xlabel("E"); plt.ylabel("I"); plt.title("E vs I")
            f2 = fig_dir / f"{tag_prefix}superposition_E_vs_I.png"
            plt.savefig(f2, dpi=dpi, bbox_inches="tight"); plt.close()
            _mirror_file(f2, mirrors, put_in_figs=True, cfg=active_cfg)

            if np.isfinite(X_arr).any():
                plt.figure(); plt.scatter(E_arr, X_arr, s=6, alpha=0.6)
                plt.xlabel("E"); plt.ylabel("X = f(E,I)"); plt.title("E vs X")
                f3 = fig_dir / f"{tag_prefix}superposition_E_vs_X.png"
                plt.savefig(f3, dpi=dpi, bbox_inches="tight"); plt.close()
                _mirror_file(f3, mirrors, put_in_figs=True, cfg=active_cfg)

    return {"dataframe": df,
            "arrays": {"E0": E_arr, "I_shannon": I_sh, "I_kl": I_kl, "I_fused": I, "X": X_arr}}


# --------------------------------------------------------------
# Wrapper for Master Controller
# --------------------------------------------------------------
def run_superposition_stage(active=None, active_cfg=None, **kwargs):
    cfg = active if active is not None else active_cfg
    if cfg is None:
        raise ValueError("Provide 'active' or 'active_cfg'")     
    return run_superposition(active_cfg=cfg, **kwargs)  
    
if __name__ == "__main__":
   run_superposition_stage(ACTIVE)
