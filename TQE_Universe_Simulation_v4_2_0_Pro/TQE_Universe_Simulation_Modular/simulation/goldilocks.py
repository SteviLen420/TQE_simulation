# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Stefan Len
#
# Goldilocks optimization module
#
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from ..config.master_ctrl import MASTER_CTRL
from ..core.pipeline_context import PipelineContext
from ..core.physics_engine import PhysicsEngine

def bayesian_adaptive_goldilocks(ctx: PipelineContext, total_budget: int = 1000):
    """
    Bayesian Adaptive Goldilocks Optimization using Gaussian Process.
    
    Intelligently samples universes in 3 iterations:
      1. Exploration: Random sampling across full X range
      2. Exploitation: Focus on high-UCB regions (likely peak areas)
      3. Refinement: Dense sampling around discovered peak
    
    Args:
        ctx: Pipeline context
        total_budget: Total number of universes to sample
    
    Returns:
        X_low, X_high, X_peak, X_peak_std (floats with uncertainty)
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
    from scipy.stats import norm
    
    # Determine I-definition name for E-only vs E+I
    if ctx.config.get("PIPELINE_VARIANT", "full") == "energy_only":
        i_def = "energy_only"
    else:
        i_def = ctx.config.get("I_DEFINITION_MODE", "kl_shannon")
    
    print(f"\n[BAYESIAN GOLDILOCKS] Starting adaptive optimization for {i_def}")
    print(f"[BAYESIAN GOLDILOCKS] Total budget: {total_budget} universes in 3 iterations")
    
    # Split budget: 30% exploration, 40% exploitation, 30% refinement
    n_explore = int(total_budget * 0.30)
    n_exploit = int(total_budget * 0.40)
    n_refine = total_budget - n_explore - n_exploit
    
    # Storage for all sampled universes
    all_X = []
    all_stability = []
    all_iterations = []  # Track which iteration each sample came from
    
    # ==================================================================
    # ITERATION 1: EXPLORATION (random sampling, wide range)
    # ==================================================================
    print(f"[BAYESIAN GOLDILOCKS] Iteration 1/3: Exploration ({n_explore} universes, random sampling)")
    
    for uid in tqdm(range(n_explore), desc="Exploring", leave=False, ncols=100):
        uni_seed = ctx.rng.integers(0, 2**31)
        uni_rng = np.random.default_rng(uni_seed)
        uni_physics = PhysicsEngine(ctx.config, uni_rng)
        
        # Sample E and I
        E = uni_physics.sample_energy(rng_local=uni_rng)
        
        if ctx.config.get("PIPELINE_VARIANT", "full") == "energy_only":
            I = 0.0
            X = E * ctx.config["X_SCALE"]
        else:
            I_defs = uni_physics.compute_all_I_definitions(E, a=1.0)
            I = I_defs.get(i_def, 0.5)
            X = uni_physics.compute_coupling(E, I)
        
        # Quick stability check
        is_stable = _check_stability_calibration(X, ctx.config, uni_rng)
        
        all_X.append(X)
        all_stability.append(float(is_stable))
        all_iterations.append(1)  # Iteration 1: Exploration
    
    # Fit initial Gaussian Process
    X_train = np.array(all_X).reshape(-1, 1)
    y_train = np.array(all_stability)
    
    # GP kernel: RBF + White noise (configurable via MASTER_CTRL)
    gp_noise = ctx.config.get("BAYESIAN_GP_NOISE", 0.01)
    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=5.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=gp_noise)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-6, normalize_y=True)
    gp.fit(X_train, y_train)
    
    # Find preliminary peak
    X_min, X_max = min(all_X), max(all_X)
    X_grid = np.linspace(X_min, X_max, 1000).reshape(-1, 1)
    mu, sigma = gp.predict(X_grid, return_std=True)
    
    preliminary_peak_idx = np.argmax(mu)
    preliminary_peak = X_grid[preliminary_peak_idx, 0]
    preliminary_std = sigma[preliminary_peak_idx]
    
    print(f"[BAYESIAN GOLDILOCKS] Preliminary peak: X ≈ {preliminary_peak:.2f} ± {preliminary_std:.2f}")
    
    # ==================================================================
    # ITERATION 2: EXPLOITATION (UCB-guided sampling)
    # ==================================================================
    print(f"[BAYESIAN GOLDILOCKS] Iteration 2/3: Exploitation ({n_exploit} universes, UCB-guided)")
    
    # Define search range (focus around preliminary peak)
    search_margin = max(10.0, preliminary_std * 3.0)
    X_search_min = max(X_min, preliminary_peak - search_margin)
    X_search_max = min(X_max, preliminary_peak + search_margin)
    
    # Get UCB kappa from config (configurable exploration-exploitation tradeoff)
    kappa = ctx.config.get("BAYESIAN_UCB_KAPPA", 2.0)
    
    for uid in tqdm(range(n_exploit), desc="Exploiting", leave=False, ncols=100):
        # UCB acquisition: sample where mu + kappa * sigma is high
        X_candidates = np.linspace(X_search_min, X_search_max, 500).reshape(-1, 1)
        mu_cand, sigma_cand = gp.predict(X_candidates, return_std=True)
        ucb = mu_cand + kappa * sigma_cand
        
        # Sample at highest UCB with some randomness
        top_k = min(10, len(ucb))
        top_indices = np.argsort(ucb)[-top_k:]
        selected_idx = ctx.rng.choice(top_indices)
        X_target = X_candidates[selected_idx, 0]
        
        # Generate universe at target X (inverse sampling)
        uni_seed = ctx.rng.integers(0, 2**31)
        uni_rng = np.random.default_rng(uni_seed)
        uni_physics = PhysicsEngine(ctx.config, uni_rng)
        
        # Sample E, compute I to get close to X_target
        # Simple approach: multiple E samples, pick closest X
        best_X, best_stable = None, None
        for _ in range(5):  # Try 5 E samples
            E = uni_physics.sample_energy(rng_local=uni_rng)
            
            if ctx.config.get("PIPELINE_VARIANT", "full") == "energy_only":
                I = 0.0
                X = E * ctx.config["X_SCALE"]
            else:
                I_defs = uni_physics.compute_all_I_definitions(E, a=1.0)
                I = I_defs.get(i_def, 0.5)
                X = uni_physics.compute_coupling(E, I)
            
            if best_X is None or abs(X - X_target) < abs(best_X - X_target):
                best_X = X
                best_stable = _check_stability_calibration(X, ctx.config, uni_rng)
        
        all_X.append(best_X)
        all_stability.append(float(best_stable))
        all_iterations.append(2)  # Iteration 2: Exploitation
        
        # Update GP
        X_train = np.array(all_X).reshape(-1, 1)
        y_train = np.array(all_stability)
        gp.fit(X_train, y_train)
    
    # Refined peak estimate
    mu_refine, sigma_refine = gp.predict(X_grid, return_std=True)
    refined_peak_idx = np.argmax(mu_refine)
    refined_peak = X_grid[refined_peak_idx, 0]
    refined_std = sigma_refine[refined_peak_idx]
    
    print(f"[BAYESIAN GOLDILOCKS] Refined peak: X ≈ {refined_peak:.2f} ± {refined_std:.2f}")
    
    # ==================================================================
    # ITERATION 3: REFINEMENT (dense sampling around peak)
    # ==================================================================
    print(f"[BAYESIAN GOLDILOCKS] Iteration 3/3: Refinement ({n_refine} universes, dense peak sampling)")
    
    # Very narrow range around refined peak
    refine_margin = max(3.0, refined_std * 2.0)
    X_refine_min = refined_peak - refine_margin
    X_refine_max = refined_peak + refine_margin
    
    for uid in tqdm(range(n_refine), desc="Refining", leave=False, ncols=100):
        # Sample uniformly in refined range
        X_target = ctx.rng.uniform(X_refine_min, X_refine_max)
        
        # Generate universe at target X
        uni_seed = ctx.rng.integers(0, 2**31)
        uni_rng = np.random.default_rng(uni_seed)
        uni_physics = PhysicsEngine(ctx.config, uni_rng)
        
        best_X, best_stable = None, None
        for _ in range(5):
            E = uni_physics.sample_energy(rng_local=uni_rng)
            
            if ctx.config.get("PIPELINE_VARIANT", "full") == "energy_only":
                I = 0.0
                X = E * ctx.config["X_SCALE"]
            else:
                I_defs = uni_physics.compute_all_I_definitions(E, a=1.0)
                I = I_defs.get(i_def, 0.5)
                X = uni_physics.compute_coupling(E, I)
            
            if best_X is None or abs(X - X_target) < abs(best_X - X_target):
                best_X = X
                best_stable = _check_stability_calibration(X, ctx.config, uni_rng)
        
        all_X.append(best_X)
        all_stability.append(float(best_stable))
        all_iterations.append(3)  # Iteration 3: Refinement
    
    # Final GP fit with all data
    X_train_final = np.array(all_X).reshape(-1, 1)
    y_train_final = np.array(all_stability)
    gp.fit(X_train_final, y_train_final)
    
    # Final prediction on fine grid
    X_grid_fine = np.linspace(min(all_X), max(all_X), 2000).reshape(-1, 1)
    mu_final, sigma_final = gp.predict(X_grid_fine, return_std=True)
    
    # Peak detection with uncertainty
    peak_idx = np.argmax(mu_final)
    X_peak = X_grid_fine[peak_idx, 0]
    X_peak_std = sigma_final[peak_idx]
    peak_stability = mu_final[peak_idx]
    
    # Goldilocks zone: half-max width
    half_max = peak_stability * 0.5
    valid_mask = mu_final >= half_max
    if np.any(valid_mask):
        valid_X = X_grid_fine[valid_mask, 0]
        X_low = valid_X.min()
        X_high = valid_X.max()
    else:
        X_low = X_peak * 0.85
        X_high = X_peak * 1.15
    
    # Create DataFrame for plotting and CSV export
    df_cal = pd.DataFrame({
        'X': all_X, 
        'stable': all_stability,
        'iteration': all_iterations,
        'iteration_name': ['exploration' if i==1 else 'exploitation' if i==2 else 'refinement' for i in all_iterations]
    })
    
    # Save Goldilocks results (CSV + PNG)
    gold_dir = os.path.join(ctx.paths["SAVE_DIR"], "Goldilocks_Results")
    os.makedirs(gold_dir, exist_ok=True)
    
    # 1. Save Bayesian calibration data CSV (sampled X, stability, iteration)
    csv_path = os.path.join(gold_dir, f"bayesian_calibration_{i_def}.csv")
    try:
        csv_path = ctx.with_variant(csv_path)
    except Exception:
        pass
    ctx.save_csv(df_cal, csv_path)
    
    # 2. Save Goldilocks plot with GP visualization
    png_path = os.path.join(gold_dir, f"goldilocks_zone_{i_def}.png")
    try:
        png_path = ctx.with_variant(png_path)
    except Exception:
        pass
    _plot_bayesian_goldilocks(df_cal, X_grid_fine.flatten(), mu_final, sigma_final, 
                              X_low, X_high, X_peak, X_peak_std, i_def, png_path, ctx.config)
    
    print(f"[BAYESIAN GOLDILOCKS] Final result: X_peak = {X_peak:.2f} ± {X_peak_std:.2f}")
    print(f"[BAYESIAN GOLDILOCKS] Goldilocks window: [{X_low:.2f}, {X_high:.2f}]")
    print(f"[BAYESIAN GOLDILOCKS] Peak stability rate: {peak_stability:.2%}")
    print(f"[BAYESIAN GOLDILOCKS] Total universes sampled: {total_budget}")
    
    # Store Goldilocks results in context for later use
    ctx.goldilocks = {
        "X_low": X_low, 
        "X_high": X_high, 
        "X_peak": X_peak, 
        "X_peak_std": X_peak_std,
        "stability_peak": peak_stability,
        "total_sampled": total_budget
    }
    
    return X_low, X_high, X_peak, X_peak_std, df_cal


def _check_stability_calibration(X, config, rng):
    """Quick stability check for Goldilocks calibration (simplified)."""
    N = config.get("CALIBRATION_EPOCHS", 500)
    eps = config.get("CALIBRATION_REL_EPS", 0.015)
    calm = config.get("CALIBRATION_CALM_STEPS", 6)
    sigma0 = config.get("CALIBRATION_NOISE_BASE", 0.20)
    
    X_curr = X
    consec = 0
    for n in range(1, N + 1):
        noise = rng.normal(0, sigma0 / np.sqrt(n))
        X_curr += noise
        delta_rel = abs(noise) / max(abs(X_curr), 1e-12)
        if delta_rel < eps:
            consec += 1
            if consec >= calm:
                return 1
        else:
            consec = 0
    return 0



def _plot_bayesian_goldilocks(df, X_grid, mu, sigma, X_low, X_high, X_peak, X_peak_std, i_def, save_path, config):
    """Plot Bayesian Adaptive Goldilocks with GP uncertainty bands."""
    
    # Create figure with extra space at bottom for legend
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # 1. Plot raw sampled points (scatter, semi-transparent)
    stable_mask = df["stable"] == 1
    unstable_mask = df["stable"] == 0
    ax.scatter(df.loc[stable_mask, "X"], df.loc[stable_mask, "stable"], 
               color='green', alpha=0.3, s=20, label='Stable universes', zorder=2)
    ax.scatter(df.loc[unstable_mask, "X"], df.loc[unstable_mask, "stable"], 
               color='red', alpha=0.3, s=20, label='Unstable universes', zorder=2)
    
    # 2. Plot GP mean prediction (thick blue line)
    ax.plot(X_grid, mu, color='#1f77b4', linewidth=3, label='GP Mean (Stability)', zorder=4)
    
    # 3. Plot GP uncertainty band (shaded, 95% confidence interval)
    ax.fill_between(X_grid, mu - 1.96*sigma, mu + 1.96*sigma, 
                    color='#1f77b4', alpha=0.2, label='95% Confidence', zorder=3)
    
    # 4. Peak marker with error bar (red) - NO LABEL (will be in info box below)
    peak_y = mu[np.argmin(np.abs(X_grid - X_peak))]
    ax.errorbar(X_peak, peak_y, xerr=X_peak_std*1.96, 
                fmt='o', color='red', markersize=14, linewidth=3, capsize=8, capthick=3,
                zorder=10)
    
    # 5. Goldilocks boundaries (green dashed lines) - NO LABEL (will be in info box below)
    ax.axvline(X_low, color='green', linestyle='--', linewidth=2.5, zorder=5, alpha=0.8)
    ax.axvline(X_high, color='green', linestyle='--', linewidth=2.5, zorder=5, alpha=0.8)
    
    # 6. Shaded Goldilocks region
    ax.axvspan(X_low, X_high, color='yellow', alpha=0.15, zorder=1)
    
    # Formatting - CLEAN TITLE with I-definition
    variant = config.get("PIPELINE_VARIANT", "full")
    if variant == "energy_only":
        title = f"Bayesian Adaptive Goldilocks Optimization - E-only"
        xlabel = "X = E"
    else:
        title = f"Bayesian Adaptive Goldilocks Optimization - {i_def}"
        xlabel = "X (E·I coupling)"
    
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel("Stability Rate", fontsize=16)
    ax.set_title(title, fontsize=18, pad=20)
    ax.legend(fontsize=11, framealpha=0.95, loc='upper right', shadow=False, ncol=1)
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=1.0)
    ax.tick_params(labelsize=13)
    ax.set_ylim([0.0, 1.05])
    
    # Add SMALL info box in BOTTOM LEFT corner (like reference goldilocks_zone plot)
    zone_width = X_high - X_low
    info_text = f"Sampled: {len(df)}\n"
    info_text += f"Peak: {X_peak:.2f} ± {X_peak_std:.2f}\n"
    info_text += f"Zone: [{X_low:.2f}, {X_high:.2f}]\n"
    info_text += f"Width: {zone_width:.2f}"
    
    # Bottom left corner position (like the other Goldilocks plot)
    ax.text(0.02, 0.02, info_text, transform=ax.transAxes, 
            fontsize=9, verticalalignment='bottom', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, pad=8),
            family='monospace')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def sigma_goldilocks(X: float, sigma0: float, alpha: float, X_c_low: float, X_c_high: float, config: dict):
    """Goldilocks-shaped noise: outside penalty + quadratic curvature inside."""
    if X_c_low is None or X_c_high is None: return sigma0
    if X < X_c_low or X > X_c_high:
        return sigma0 * config["OUTSIDE_PENALTY"]
    mid = 0.5 * (X_c_low + X_c_high)
    width = max(0.5 * (X_c_high - X_c_low), 1e-12)
    dist = abs(X - mid) / width
    return sigma0 * (1 + alpha * dist**2)


def simulate_lock_in(X: float, N_epoch: int, config: dict, sigma0: float, alpha: float, X_c_low: float = None, X_c_high: float = None, rng: np.random.Generator = None, E: float = None, physics_engine = None):
    """
    TQE-CONSISTENT lock-in simulation.
    
    CRITICAL CHANGE: Lock-in ONLY tracks X (E+I coupling) stability!
    - E-only mode: X = E × X_SCALE (constant) → no lock-in possible (E doesn't change)
    - E+I mode: X = f(E, I) via compute_coupling() → lock-in when ΔX stabilizes
      Supported X_MODE:
        • "E_plus_I": X = (E + α×I) × scale
        • "product": X = E × α × I × scale
        • "E_times_I_pow": X = E × (α×I)^power × scale
    
    A, ns, H are NOT tracked here! They are EMERGENT from X (computed post-lock-in).
    
    TQE Philosophy:
      E (energy) = PRIMARY
      I (information) = E's internal property
      X = f(E, I) = Universe foundation (computed via compute_coupling)
      
      Everything else (A, ns, H, CMB, anomalies, laws) → EMERGES from X
      Lock-in = X stabilization (energetic state locked)
    """
    from collections import deque
    if rng is None: rng = np.random.default_rng()

    stable_at, lockin_at, consec_stable, consec_lockin = None, None, 0, 0
    window = deque(maxlen=config["LOCKIN_WINDOW"])
    _eps = 1e-12
    
    # Track X (E+I coupling) evolution
    X_prev = None
    is_eonly_mode = (config.get("PIPELINE_VARIANT", "full") == "energy_only")

    def _agg(vals):
        m = config["LOCKIN_ROLL_METRIC"]
        if m == "median": return float(np.median(vals))
        if m == "max": return float(np.max(vals))
        return float(np.mean(vals))

    # LOCK-IN DETECTION MODE:
    # - E-only: Track emergent properties (A, ns, H) for lock-in detection
    # - E+I: Track X (E+I coupling) for TQE-consistent lock-in detection
    
    # Initialize emergent properties (for E-only mode)
    A, ns, H = rng.normal(50, 5), rng.normal(0.8, 0.05), rng.normal(0.7, 0.08)

    for n in range(1, N_epoch + 1):
        # Calculate noise level (Goldilocks-modulated, time-decaying)
        sigma = sigma_goldilocks(X, sigma0, alpha, X_c_low, X_c_high, config)
        decay = (config["NOISE_FLOOR_FRAC"] + (1 - config["NOISE_FLOOR_FRAC"]) * np.exp(-n / config["NOISE_DECAY_TAU"]))
        sigma = max(config["LL_BASE_NOISE"], sigma * decay)

        if is_eonly_mode:
            # E-only: Track emergent properties (A, ns, H) → REALISTIC lock-in rate
            A_prev, ns_prev, H_prev = A, ns, H
            A  += rng.normal(0, sigma * config["NOISE_COEFF_A"])
            ns += rng.normal(0, sigma * config["NOISE_COEFF_NS"])
            H  += rng.normal(0, sigma * config["NOISE_COEFF_H"])
            
            # Delta calculation (emergent properties)
            delta_rel = (abs(A - A_prev) / max(abs(A_prev), _eps) +
                        abs(ns - ns_prev) / max(abs(ns_prev), _eps) +
                        abs(H - H_prev) / max(abs(H_prev), _eps)) / 3.0
        else:
            # E+I mode: Track X (E+I coupling) → TQE-CONSISTENT
            if E is not None and physics_engine is not None:
                i_def_mode = config.get("I_DEFINITION_MODE", "kl_shannon")
                
                # All I-definitions use stochastic computation
                I_current = physics_engine.compute_all_I_definitions(E, a=1.0).get(i_def_mode, 0.5)
                
                # FIXED: Use compute_coupling for consistency with initial X computation!
                X_current = physics_engine.compute_coupling(E, I_current)
            else:
                # Fallback: X constant (no physics engine)
                X_current = X
            
            # Calculate ΔX (TQE-consistent: ONLY X matters!)
            if X_prev is not None:
                delta_rel = abs(X_current - X_prev) / max(abs(X_prev), _eps)
            else:
                delta_rel = 0.0  # First epoch
            
            X_prev = X_current
        
        window.append(delta_rel)

        # Stability check
        if delta_rel < config["REL_EPS_STABLE"]:
            consec_stable += 1
            if consec_stable >= config["CALM_STEPS_STABLE"] and stable_at is None: 
                stable_at = n
        else:
            consec_stable = 0

        # Lock-in check
        can_check_lock = (len(window) == window.maxlen) and (n >= config["MIN_LOCKIN_EPOCH"])
        if config["LOCKIN_REQUIRES_STABLE"]: 
            can_check_lock = can_check_lock and (stable_at is not None)
        if config["LOCKIN_MIN_STABLE_EPOCH"] > 0 and stable_at is not None:
             can_check_lock = can_check_lock and (n - stable_at >= config["LOCKIN_MIN_STABLE_EPOCH"])

        # Get I-definition specific lock-in threshold
        i_def_mode = config.get("I_DEFINITION_MODE", "kl_shannon")
        i_def_thresholds = config.get("I_DEFINITION_LOCKIN_THRESHOLDS", {})
        lockin_threshold = i_def_thresholds.get(i_def_mode, config["REL_EPS_LOCKIN"])
        
        if can_check_lock and (_agg(window) < lockin_threshold):
            consec_lockin += 1
            if consec_lockin >= config["CALM_STEPS_LOCKIN"] and lockin_at is None: 
                lockin_at = n
        else:
            consec_lockin = 0

    is_stable = 1 if stable_at is not None else 0
    is_lockin = 1 if lockin_at is not None else 0
    return is_stable, is_lockin, (stable_at if stable_at else -1), (lockin_at if lockin_at else -1)


def compute_dynamic_goldilocks(df_in: pd.DataFrame, config: dict, score_col: str = "stable") -> tuple:
    """
    Estimate Goldilocks window dynamically from a curve P(score_col | X).
    
    FIX #3: Adaptive bin count, proper sorting, and X normalization.
    """
    if df_in is None or len(df_in) == 0:
        return None, None, np.array([]), np.array([]), np.array([]), np.array([]), df_in

    Xvals = pd.to_numeric(df_in["X"], errors="coerce").values
    if np.all(~np.isfinite(Xvals)):
        return None, None, np.array([]), np.array([]), np.array([]), np.array([]), df_in

    # FIX #3a: Adaptive bin count based on sample size (avoid over-binning with small datasets)
    n_samples = len(df_in)
    nbins_adaptive = int(min(max(10, n_samples // 50), config.get("STAB_BINS", 40)))
    nbins = nbins_adaptive
    min_per_bin = int(max(1, config.get("STAB_MIN_COUNT", 10)))

    x_min, x_max = np.nanmin(Xvals), np.nanmax(Xvals)
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_min == x_max:
        return None, None, np.array([]), np.array([]), np.array([]), np.array([]), df_in

    bins = np.linspace(x_min, x_max, nbins + 1)
    df_tmp = df_in.copy()
    idx = np.digitize(df_tmp["X"].values, bins, right=False)
    idx[idx == 0] = 1
    df_tmp["bin"] = idx
    df_tmp = df_tmp[(df_tmp["bin"] > 0) & (df_tmp["bin"] <= nbins)]

    bin_stats = df_tmp.groupby("bin").agg(
        mean_X=("X", "mean"),
        score_rate=(score_col, "mean"),
        count=(score_col, "size")
    ).dropna()
    bin_stats = bin_stats[bin_stats["count"] >= min_per_bin]
    if bin_stats.empty:
        return None, None, np.array([]), np.array([]), np.array([]), np.array([]), df_tmp

    # FIX #3b: Ensure proper sorting and remove duplicates
    bin_stats = bin_stats.sort_values("mean_X").reset_index(drop=True)
    xx, yy = bin_stats["mean_X"].values, np.clip(bin_stats["score_rate"].values, 0.0, 1.0)

    # Remove duplicate X values (average Y if duplicates exist)
    if len(xx) > 1:
        df_u = pd.DataFrame({"x": xx, "y": yy}).groupby("x", as_index=False)["y"].mean()
        xx, yy = df_u["x"].values, df_u["y"].values
    
    # FIX #3c: Ensure xx is strictly sorted (critical for spline fitting)
    sort_idx = np.argsort(xx)
    xx, yy = xx[sort_idx], yy[sort_idx]
    
    # Generate smooth interpolation grid
    xs = np.linspace(xx.min(), xx.max(), 300)
    
    if len(xx) >= 2:
        k_max = max(1, len(xx) - 1)
        k_use = min(config.get("SPLINE_K", 3), k_max)
        try:
            if k_use >= 2:
                # Use smaller smoothing for sharper peak (matching reference image)
                from scipy.interpolate import UnivariateSpline
                # s=0.01 gives sharp peak like reference, not over-smoothed
                spline = UnivariateSpline(xx, yy, k=k_use, s=0.01)
                ys = spline(xs)
            else:
                ys = np.interp(xs, xx, yy)
        except Exception:
            ys = np.interp(xs, xx, yy)
    else:
        xs, ys = xx.copy(), yy.copy()

    if score_col == "stable": ys = np.clip(ys, 0.0, 1.0)

    if len(xs) == 0 or len(ys) == 0:
        return None, None, xs, ys, xx, yy, df_tmp

    peak_idx = int(np.argmax(ys))
    peak_val = float(ys[peak_idx])

    threshold = float(config.get("GOLDILOCKS_THRESHOLD", 0.5))
    half_max = threshold * peak_val

    if not np.isfinite(peak_val) or peak_val <= 1e-12:
        margin = float(config.get("GOLDILOCKS_MARGIN", 0.10))
        x_mid = float(np.median(xx)) if len(xx) else float(np.median(Xvals))
        X_c_low = x_mid * (1 - margin)
        X_c_high = x_mid * (1 + margin)
        return X_c_low, X_c_high, xs, ys, xx, yy, df_tmp

    valid_mask = ys >= half_max
    if np.any(valid_mask):
        valid_region = xs[valid_mask]
        X_c_low = float(valid_region.min())
        X_c_high = float(valid_region.max())
    else:
        peak_x = float(xs[peak_idx])
        margin = float(config.get("GOLDILOCKS_MARGIN", 0.10))
        X_c_low = peak_x * (1 - margin)
        X_c_high = peak_x * (1 + margin)

    return X_c_low, X_c_high, xs, ys, xx, yy, df_tmp


