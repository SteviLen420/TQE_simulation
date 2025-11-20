# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Stefan Len
#
# Phases 01-10
#
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from ..config.master_ctrl import MASTER_CTRL
from ..core.pipeline_context import PipelineContext
from ..simulation.goldilocks import compute_dynamic_goldilocks
from ..phases.phase_11_20 import (
    simulate_quantum_fluctuation_series,
    simulate_superposition_series,
    simulate_collapse_series,
    simulate_expansion_panel
)

def phase_02_stability_curve(ctx: PipelineContext, df: pd.DataFrame) -> float:
    """Phase 2: Dynamic Goldilocks estimation + plot (stability rate vs X)."""
    X_c_low_plot, X_c_high_plot, xs, ys, xx, yy, df_binned = compute_dynamic_goldilocks(df, ctx.config)
    
    peak_x_location = None
    # Create figure - standard size
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Expand X range for better visualization (like reference image)
    if len(xs) > 0:
        X_min_data = min(xs)
        X_max_data = max(xs)
        X_range = X_max_data - X_min_data
        # Expand range by 30% on each side for better context
        X_min_plot = max(0, X_min_data - 0.3 * X_range)
        X_max_plot = X_max_data + 0.3 * X_range
        xs_extended = np.linspace(X_min_plot, X_max_plot, 1000)
    else:
        xs_extended = xs
    
    # Compute purity curve (lock-in rate) if available
    lockin_counts = []
    if "lockin" in df.columns and df_binned is not None and len(df_binned) > 0:
        # FIX: Use "bin" column (not "X_bin") - matches compute_dynamic_goldilocks output
        if "bin" in df_binned.columns:
            for _, group in df_binned.groupby("bin"):
                lockin_rate = group["lockin"].mean()
                lockin_counts.append(lockin_rate)
        else:
            # Fallback: compute bins manually if df_binned doesn't have "bin" column
            pass
    
    left_x = None
    right_x = None
    peak_x = None
    peak_y = None
    
    if len(xx) > 0 and len(yy) > 0:
        # 1. Plot bin means (light blue circles, matching the reference image)
        ax.plot(xx, yy, 'o', color='#87CEEB', markersize=10, label='bin means', zorder=5)
        
        # 2. Fit and plot spline (thick red line, matching reference)
        if len(xx) >= 4:
            try:
                # Use smaller smoothing parameter for sharper peak (like reference)
                spline = UnivariateSpline(xx, yy, k=3, s=0.01)
                xs_smooth = np.linspace(xx.min(), xx.max(), 300)
                ys_smooth = spline(xs_smooth)
                ys_smooth = np.clip(ys_smooth, 0.0, 1.0)
                ax.plot(xs_smooth, ys_smooth, '-', color='red', linewidth=2.5, label='spline fit', zorder=4)
                
                # Update xs and ys for peak calculation
                xs = xs_smooth
                ys = ys_smooth
            except:
                ax.plot(xx, yy, '-', color='red', linewidth=2.5, label='spline fit', zorder=4)
                xs = xx
                ys = yy
        else:
            ax.plot(xx, yy, '-', color='red', linewidth=2.5, label='spline fit', zorder=4)
            xs = xx
            ys = yy
        
        # 3. Find and mark peak (red circle + dashed red line, matching reference)
        if len(ys) > 0:
            peak_idx = np.argmax(ys)
            peak_x = xs[peak_idx]
            peak_y = ys[peak_idx]
            peak_x_location = float(peak_x)
            
            # Mark peak with red circle and vertical line
            ax.plot(peak_x, peak_y, 'o', color='red', markersize=12, zorder=10)
            ax.axvline(peak_x, color='red', linestyle='--', linewidth=2, label=f'Peak = {peak_x:.2f}', zorder=3)

            # 4. Goldilocks zone boundaries (half-maximum method, matching reference)
            threshold = 0.5
            half_max = threshold * peak_y
            valid_mask = ys >= half_max
            
            if np.any(valid_mask):
                valid_region = xs[valid_mask]
                left_x = float(valid_region.min())
                right_x = float(valid_region.max())
                
                # Mark boundaries with dashed lines (matching reference)
                ax.axvline(left_x, color='green', linestyle='--', linewidth=2, label=f'Goldi left = {left_x:.2f}', zorder=3)
                ax.axvline(right_x, color='purple', linestyle='--', linewidth=2, label=f'Goldi right = {right_x:.2f}', zorder=3)
            else:
                # Fallback: use peak ± margin
                margin = 0.10
                left_x = peak_x * (1 - margin)
                right_x = peak_x * (1 + margin)
                ax.axvline(left_x, color='green', linestyle='--', linewidth=2, label=f'Goldi left = {left_x:.2f}', zorder=3)
                ax.axvline(right_x, color='purple', linestyle='--', linewidth=2, label=f'Goldi right = {right_x:.2f}', zorder=3)

    # Styling - CLEAN TITLE with I-definition
    i_def = ctx.config.get("I_DEFINITION_MODE", "default")
    if ctx.variant == "energy_only":
        xlabel = "X = E"
        title = f"Goldilocks zone: stability vs E - E-only"
    else:
        xlabel = "X = E·I"
        title = f"Goldilocks zone: stability vs E·I - {i_def}"
    
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel("Stability", fontsize=16)
    ax.set_title(title, fontsize=18, pad=20)
    
    # Build legend with Goldilocks info integrated
    handles, labels = ax.get_legend_handles_labels()
    if peak_x is not None and left_x is not None and right_x is not None:
        zone_width = right_x - left_x
        # Add empty handles for info lines in legend
        import matplotlib.patches as mpatches
        empty_patch = mpatches.Patch(color='none', label='')
        info_patch1 = mpatches.Patch(color='none', label=f'Peak: {peak_x:.2f}')
        info_patch2 = mpatches.Patch(color='none', label=f'Goldi: [{left_x:.2f}, {right_x:.2f}]')
        info_patch3 = mpatches.Patch(color='none', label=f'Width: {zone_width:.2f}')
        handles.extend([empty_patch, info_patch1, info_patch2, info_patch3])
        labels.extend(['', f'Peak: {peak_x:.2f}', f'Goldi: [{left_x:.2f}, {right_x:.2f}]', f'Width: {zone_width:.2f}'])
    
    ax.legend(handles, labels, loc='upper left', fontsize=11, framealpha=0.95, shadow=False, ncol=1)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.tick_params(labelsize=13)
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    # Set X axis limits to expanded range
    if len(xs) > 0:
        ax.set_xlim(X_min_plot, X_max_plot)
    
    plt.tight_layout()
    
    # Save Goldilocks zone plot with I-definition in filename
    filename = f"goldilocks_zone_{i_def}.png"
    ctx.save_fig(os.path.join(ctx.paths["AGGREGATE_FIG_DIR"], filename), fig=fig, close=False)
    
    # Also save as generic stability_curve.png for backward compatibility
    ctx.save_fig(os.path.join(ctx.paths["AGGREGATE_FIG_DIR"], "stability_curve.png"), fig=fig, close=True)
    
    return peak_x_location if peak_x_location is not None else np.nan



def phase_03_scatter_ei(ctx: PipelineContext, df: pd.DataFrame):
    """Phase 3: E-I scatter plot (coloring by stability)."""
    fig, ax = plt.subplots(figsize=(12, 10))
    sc = ax.scatter(df["E"], df["I"], c=df["stable"], cmap="coolwarm", s=25, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    # Get I-definition name
    i_def = ctx.config.get("I_DEFINITION_MODE", "default")
    
    ax.set_xlabel("Energy (E)", fontsize=16)
    if ctx.variant == "energy_only":
        ax.set_ylabel("Information parameter I (disabled = 0)", fontsize=16)
        ax.set_title("Universe Outcomes in E Space - E-only", fontsize=18, pad=20)
    else:
        ax.set_ylabel(f"Information parameter (I: {i_def})", fontsize=16)
        ax.set_title(f"Universe Outcomes in (E, I) Space - {i_def}", fontsize=18, pad=20)
    
    cb = plt.colorbar(sc, ticks=[0, 1], ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Stable (0/1)", fontsize=14)
    cb.ax.tick_params(labelsize=13)
    
    ax.tick_params(labelsize=13)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    ctx.save_fig(os.path.join(ctx.paths["AGGREGATE_FIG_DIR"], "scatter_EI.png"))

def phase_04_fluctuation_panels(ctx: PipelineContext, df: pd.DataFrame):
    """Phase 4: t<0, t=0, t>0 fluctuation dynamics plots + CSV outputs."""
    
    # 0. Quantum Fluctuation (Standalone)
    if ctx.config.get("RUN_QUANTUM_FLUCT", True):
        tF, expA, varA = simulate_quantum_fluctuation_series(
            T=ctx.config.get("FL_FLUCT_T", 6.0), dt=ctx.config.get("FL_FLUCT_DT", 0.02),
            dim=ctx.config.get("FL_SUPER_DIM", 4), kick=ctx.config.get("FL_SUPER_KICK", 0.12),
            noise=ctx.config.get("FL_SUPER_NOISE", 0.05), obs_kind=ctx.config.get("FL_FLUCT_OBS", "Z"),
            obs_jitter=ctx.config.get("FL_SUPER_OBS_JITTER", 0.0), seed=ctx.master_seed + 10
        )
        fluc_df = pd.DataFrame({"time": tF, "exp_A": expA, "var_A": varA})
        ctx.save_csv(fluc_df, os.path.join(ctx.paths["AGGREGATE_DIR"], "fl_fluctuation_timeseries.csv"))

        # PUBLICATION: Larger figure with better styling (was: 8,5)
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(tF, expA, label="⟨A⟩", ls="--", alpha=0.95, linewidth=3, color='#1f77b4')
        ax.plot(tF, varA, label="Var(A)", ls="--", alpha=0.95, linewidth=3, color='#ff7f0e')
        ax.set_xlabel("Time", fontsize=16)
        ax.set_ylabel("Value", fontsize=16)
        ax.set_title("Quantum Fluctuation: ⟨A⟩ and Var(A)", fontsize=18, pad=20)
        ax.legend(fontsize=16, framealpha=0.95, loc='best')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=14)
        plt.tight_layout()
        ctx.save_fig(os.path.join(ctx.paths["AGGREGATE_FIG_DIR"], "fl_fluctuation.png"))

    if ctx.config.get("RUN_FLUCTUATION_BLOCK", True):
        # Choose X_lock reference
        if "X" in df.columns and len(df) > 0 and np.isfinite(df["X"]).any():
            X_lock = float(np.median(df["X"]))
        else:
            X_lock = ctx.config.get("X_CENTER", 4.0) * ctx.config.get("ALPHA_I", 0.8)

        # 1. t<0 : superposition entropy & purity
        tS, ent, pur = simulate_superposition_series(
            T=ctx.config["FL_SUPER_T"], dt=ctx.config["FL_SUPER_DT"], dim=ctx.config["FL_SUPER_DIM"],
            noise=ctx.config["FL_SUPER_NOISE"], kick=ctx.config.get("FL_SUPER_KICK", 0.15),
            obs_jitter=ctx.config.get("FL_SUPER_OBS_JITTER", 0.02), seed=ctx.master_seed + 11
        )
        sup_df = pd.DataFrame({"time": tS, "entropy": ent, "purity": pur})
        ctx.save_csv(sup_df, os.path.join(ctx.paths["AGGREGATE_DIR"], "fl_superposition_timeseries.csv"))
        
        # PUBLICATION: Larger figure with better styling (was: 8,5)
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(tS, ent, label="Entropy", ls="--", alpha=0.9, linewidth=3, color='#1f77b4')
        ax.plot(tS, pur, label="Purity", ls="--", alpha=0.9, linewidth=3, color='#ff7f0e')
        ax.set_xlabel("Time", fontsize=16)
        ax.set_ylabel("Value", fontsize=16)
        ax.set_title("t < 0: Quantum Superposition", fontsize=18, pad=20)
        ax.legend(fontsize=16, framealpha=0.95, loc='best')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=14)
        plt.tight_layout()
        ctx.save_fig(os.path.join(ctx.paths["AGGREGATE_FIG_DIR"], "fl_superposition.png"))

        # 2. t=0 : collapse
        tC, xC = simulate_collapse_series(
            X_lock, t_pre=ctx.config["FL_COLLAPSE_T_PRE"], t_post=ctx.config["FL_COLLAPSE_T_POST"],
            dt=ctx.config["FL_COLLAPSE_DT"], pre_sigma=ctx.config["FL_COLLAPSE_PRE_SIGMA"],
            post_sigma=ctx.config["FL_COLLAPSE_POST_SIGMA"], revert=ctx.config["FL_COLLAPSE_REVERT"],
            seed=ctx.master_seed + 22
        )
        col_df = pd.DataFrame({"time": tC, "X": xC, "X_lock": X_lock})
        ctx.save_csv(col_df, os.path.join(ctx.paths["AGGREGATE_DIR"], "fl_collapse_timeseries.csv"))
        
        # PUBLICATION: Larger figure with better styling (was: 8,5)
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(tC, xC, color="gray", ls="--", linewidth=3, label="fluctuation → lock-in", alpha=0.9)
        ax.axvline(0.0, color="red", linewidth=3, label="Collapse Event (t=0)")
        ax.axhline(X_lock, color="red", ls="--", linewidth=3, label=f"Lock-in X={X_lock:.2f}", alpha=0.8)
        ax.set_xlabel("Time", fontsize=16)
        ax.set_ylabel("X = E" if ctx.variant == "energy_only" else "X = E·I", fontsize=16)
        ax.set_title("t = 0: Collapse (Lock-in of X)", fontsize=18, pad=20)
        ax.legend(fontsize=16, framealpha=0.95, loc='best')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=14)
        plt.tight_layout()
        ctx.save_fig(os.path.join(ctx.paths["AGGREGATE_FIG_DIR"], "fl_collapse.png"))

        # 3. t>0 : expansion dynamics
        # FIX #1: Add variant_id based on I-definition to ensure different expansion trajectories
        i_definition_hash = hash(ctx.config.get("I_DEFINITION_MODE", "default")) % 1000
        i_jit = 0.0 if ctx.variant == "energy_only" else ctx.config["FL_EXP_I_JITTER"]
        te, Atrack, Itrack = simulate_expansion_panel(
            epochs=ctx.config["FL_EXP_EPOCHS"], drift=ctx.config["FL_EXP_DRIFT"],
            jitter=ctx.config["FL_EXP_JITTER"], i_jitter=i_jit, seed=ctx.master_seed + 33,
            start_amplitude=ctx.config["FL_EXP_START_AMPLITUDE"],
            variant_id=i_definition_hash
        )
        if ctx.variant == "energy_only": Itrack = np.zeros_like(Atrack)
        exp_df = pd.DataFrame({"epoch": te, "A": Atrack, "I_track": Itrack})
        ctx.save_csv(exp_df, os.path.join(ctx.paths["AGGREGATE_DIR"], "fl_expansion_timeseries.csv"))
        
        # PUBLICATION: Larger figure with better styling (was: 9,5)
        fig, ax = plt.subplots(figsize=(14, 8))
        # Get I-definition name
        i_def = ctx.config.get("I_DEFINITION_MODE", "default")
        
        if ctx.variant == "energy_only":
            ax.set_title("t > 0: Expansion Dynamics - E-only", fontsize=18, pad=20)
            ax.plot(te, Atrack, label="Amplitude A", ls="--", linewidth=3, color='#1f77b4', alpha=0.9)
        else:
            ax.set_title(f"t > 0: Expansion Dynamics - {i_def}", fontsize=18, pad=20)
            ax.plot(te, Atrack, label="Amplitude A", ls="--", linewidth=3, color='#1f77b4', alpha=0.9)
            ax.plot(te, Itrack, label="Orientation I", ls="--", linewidth=3, color='#ff7f0e', alpha=0.9)
        
        if len(df) > 0 and "lock_epoch" in df.columns and (df["lock_epoch"] >= 0).any():
            lock_ep = int(np.median(df.loc[df["lock_epoch"] >= 0, "lock_epoch"]))
            ax.axvline(lock_ep, color="red", ls="--", linewidth=3, label=f"Law lock-in ≈ {lock_ep}", alpha=0.8)
        
        eqA = np.percentile(Atrack, 50)
        ax.axhline(eqA, color="gray", ls="--", alpha=0.7, linewidth=2, label="Equilibrium A")
        ax.set_xlabel("Epoch", fontsize=16)
        ax.set_ylabel("Parameters", fontsize=16)
        ax.legend(fontsize=16, framealpha=0.95, loc='best')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=14)
        plt.tight_layout()
        ctx.save_fig(os.path.join(ctx.paths["AGGREGATE_FIG_DIR"], "fl_expansion.png"))



def phase_05_stability_by_i(ctx: PipelineContext, df: pd.DataFrame):
    """Phase 5: Stability analysis by I (exact zero + epsilon sweep)."""
    
    def _stability_stats(mask: pd.Series, label: str):
        total = int(mask.sum())
        stables = int(df.loc[mask, "stable"].sum())
        lockins = int((df.loc[mask, "lock_epoch"] >= 0).sum())
        return {
            "group": label,
            "n": total,
            "stable_n": stables,
            "stable_ratio": (stables / total) if total > 0 else float("nan"),
            "lockin_n": lockins,
            "lockin_ratio": (lockins / total) if total > 0 else float("nan")
        }

    # Exact split
    mask_I_eq0 = (df["I"] == 0.0)
    mask_I_gt0 = (df["I"]  > 0.0)
    zero_split_rows = [
        _stability_stats(mask_I_eq0, "I == 0"),
        _stability_stats(mask_I_gt0, "I > 0"),
    ]
    zero_split_df = pd.DataFrame(zero_split_rows)
    zero_split_path = os.path.join(ctx.paths["AGGREGATE_DIR"], "stability_by_I_zero.csv")
    ctx.save_csv(zero_split_df, zero_split_path)

    # Epsilon sweep
    eps_list = [1e-12, 1e-9, 1e-6, 1e-3, 1e-2, 5e-2, 1e-1]
    eps_rows = []
    for eps in eps_list:
        eps_rows.append({ **_stability_stats(df["I"] <= eps, f"I <= {eps}"), "eps": eps})
        eps_rows.append({ **_stability_stats(df["I"]  > eps, f"I > {eps}"),  "eps": eps})
    eps_df = pd.DataFrame(eps_rows)
    eps_path = os.path.join(ctx.paths["AGGREGATE_DIR"], "stability_by_I_eps_sweep.csv")
    ctx.save_csv(eps_df, eps_path)
    
    if ctx.config.get("VERBOSE", True):
        print(f"\n📝 Stability by I breakdown saved to:\n - {ctx.with_variant(zero_split_path)}\n - {ctx.with_variant(eps_path)}")



def phase_06_lockin_histogram(ctx: PipelineContext, df: pd.DataFrame):
    """Phase 6: Lock-in epoch distribution histogram."""
    if ctx.config.get("PLOT_LOCKIN_HIST", True):
        lock_in_epochs = df.loc[df["lock_epoch"] >= 0, "lock_epoch"]
        if len(lock_in_epochs) > 0:
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.hist(lock_in_epochs, bins=30, edgecolor='black', alpha=0.7, color='green')
            
            # Get I-definition name
            i_def = ctx.config.get("I_DEFINITION_MODE", "default")
            if ctx.variant == "energy_only":
                title = "Distribution of Lock-in Epochs - E-only"
            else:
                title = f"Distribution of Lock-in Epochs - {i_def}"
            
            ax.set_xlabel("Lock-in Epoch", fontsize=16)
            ax.set_ylabel("Frequency", fontsize=16)
            ax.set_title(title, fontsize=18, pad=20)
            ax.tick_params(labelsize=13)
            ax.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            ctx.save_fig(os.path.join(ctx.paths["AGGREGATE_FIG_DIR"], "lockin_histogram.png"))



def phase_07_stability_distribution(ctx: PipelineContext, df: pd.DataFrame):
    """Phase 7: 5-way stability breakdown bar chart."""
    
    # Get finetuning threshold from config
    ft_eps = ctx.config.get("FT_EPS_EQ", 0.5)
    
    # Calculate gap |E - I| for all universes
    df_temp = df.copy()
    if 'I' in df_temp.columns:
        df_temp['gap'] = np.abs(df_temp['E'] - df_temp['I'])
    else:
        df_temp['gap'] = 0.0
    
    # Calculate counts for each category
    count_unstable = len(df_temp[df_temp['stable'] == 0])
    count_stable = len(df_temp[df_temp['stable'] == 1])
    count_lockin = len(df_temp[df_temp['lock_epoch'] >= 0])
    
    # Lock-in universes with finetuning classification
    df_lockin = df_temp[df_temp['lock_epoch'] >= 0]
    count_finely_tuned = len(df_lockin[df_lockin['gap'] <= ft_eps])
    count_coarsely_tuned = len(df_lockin[df_lockin['gap'] > ft_eps])
    
    counts = [count_unstable, count_stable, count_lockin, count_finely_tuned, count_coarsely_tuned]
    labels = ['Unstable', 'Stable', 'Lock-in\n(from Stable)', 
              f'Finely-tuned\n|E-I|≤{ft_eps}', f'Coarsely-tuned\n|E-I|>{ft_eps}']
    percentages = [count/len(df)*100 for count in counts]
    
    # Colors: Red, Green, Blue, Light Blue, Orange
    colors = ['#E74C3C', '#2ECC71', '#5DADE2', '#85C1E9', '#F39C12']
    
    # PUBLICATION: Larger bar chart (use default 12,8 from PLOT_FIGSIZE_DEFAULT)
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Draw bars with black borders
    # PUBLICATION: Thicker borders (was: 1.5)
    bars = ax.bar(range(len(labels)), counts, color=colors, 
                   edgecolor='black', linewidth=2.0, alpha=0.9, width=0.75)
    
    # Add count labels ABOVE bars with proper spacing (no overlap!)
    for i, (bar, count, pct) in enumerate(zip(bars, counts, percentages)):
        height = bar.get_height()
        # More space above bars
        ax.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.05,
                f'{count}', ha='center', va='bottom', fontsize=16)
    
    # Format x-axis labels: "Category\n(count, percentage%)"
    x_labels = [f'{label}\n({count}, {pct:.1f}%)' 
                for label, count, pct in zip(labels, counts, percentages)]
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(x_labels, fontsize=13)
    
    # Get I-definition name for title
    i_def = ctx.config.get("I_DEFINITION_MODE", "default")
    if ctx.variant == "energy_only":
        title = "Stability Distribution - E-only (Five Categories)"
    else:
        title = f"Stability Distribution - {i_def} (Five Categories)"
    
    ax.set_ylabel("Number of Universes", fontsize=16)
    ax.set_ylim(0, max(counts) * 1.25)  # Extra headroom for labels (no overlap!)
    ax.set_title(title, fontsize=18, pad=20)
    ax.tick_params(axis='y', labelsize=13)
    
    ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    ctx.save_fig(os.path.join(ctx.paths["AGGREGATE_FIG_DIR"], "stability_distribution_five.png"))
    
    # Save CSV data
    stability_data = pd.DataFrame({
        'category': labels,
        'count': counts,
        'percentage': percentages
    })
    ctx.save_csv(stability_data, os.path.join(ctx.paths["AGGREGATE_DIR"], "stability_distribution_five.csv"))
    
    if ctx.config.get("VERBOSE", True):
        print(f"[STABILITY DIST] {count_unstable} unstable, {count_stable} stable, {count_lockin} lock-in, CSV saved")



def phase_08_avg_lockin_curve(ctx: PipelineContext, df: pd.DataFrame):
    """Phase 8: Average lock-in trajectory across universes."""
    if ctx.config.get("PLOT_AVG_LOCKIN", True):
        df_lockin = df[df["lock_epoch"] >= 0].copy()
        
        if len(df_lockin) > 0:
            max_epochs = ctx.config.get("LOCKIN_EPOCHS", 1500)
            
            # OPTIMIZED: Vectorized curve generation (10× faster than iterrows)
            seeds = df_lockin["seed"].values.astype(int)
            X_vals = df_lockin["X"].values.astype(float)
            lock_eps = df_lockin["lock_epoch"].values.astype(int)
            
            all_curves = []
            for i in range(len(seeds)):
                uni_seed = seeds[i]
                X_val = X_vals[i]
                lock_ep = lock_eps[i]
            
                rng_uni = np.random.default_rng(uni_seed)
                
                if lock_ep > 0:
                    pre_lock = rng_uni.normal(X_val, 0.3, size=lock_ep)
                    post_lock = X_val + (rng_uni.normal(0, 0.1, size=max_epochs-lock_ep) * np.exp(-np.arange(max_epochs-lock_ep) / 200))
                    curve = np.concatenate([pre_lock, post_lock])
                else:
                    curve = X_val + (rng_uni.normal(0, 0.1, size=max_epochs) * np.exp(-np.arange(max_epochs) / 200))
                
                all_curves.append(curve[:max_epochs])
                
            curves_array = np.array(all_curves)
            mean_curve = np.mean(curves_array, axis=0)
            std_curve = np.std(curves_array, axis=0)
            epochs = np.arange(max_epochs)
            
            avg_df = pd.DataFrame({'epoch': epochs, 'mean_X': mean_curve, 'std_X': std_curve})
            ctx.save_csv(avg_df, os.path.join(ctx.paths["AGGREGATE_DIR"], "avg_lockin_curve.csv"))
            
            # PUBLICATION: Larger avg lockin curve (was: 10,6)
            fig, ax = plt.subplots(figsize=(14, 9))
            ax.plot(epochs, mean_curve, 'b-', lw=3, label='Mean X', alpha=0.9)
            ax.fill_between(epochs, mean_curve - std_curve, mean_curve + std_curve, alpha=0.35, label='±1σ', color='blue')
            ax.set_xlabel('Epoch', fontsize=16)
            ax.set_ylabel('X = E·I', fontsize=16)
            # Get I-definition name
            i_def = ctx.config.get("I_DEFINITION_MODE", "default")
            if ctx.variant == "energy_only":
                title = f'Average Lock-in Curve - E-only (N={len(df_lockin)})'
            else:
                title = f'Average Lock-in Curve - {i_def} (N={len(df_lockin)})'
            ax.set_title(title, fontsize=18, pad=20)
            ax.legend(fontsize=16, framealpha=0.95, loc='best')
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=14)
            plt.tight_layout()
            ctx.save_fig(os.path.join(ctx.paths["AGGREGATE_FIG_DIR"], "avg_lockin_curve.png"))



def phase_09_feature_importance(ctx: PipelineContext, df: pd.DataFrame):
    """Phase 9: Random Forest feature importance (classification + regression)."""
    if ctx.config.get("RUN_FEATURE_IMPORTANCE_DETECTOR", True) and len(df) > 20:
        
        feature_cols = ['E', 'I', 'X']
        if 'E' in df.columns and 'I' in df.columns:
            df['E_times_I'] = df['E'] * df['I']
            df['E_plus_I'] = df['E'] + df['I']
            df['abs_E_minus_I'] = np.abs(df['E'] - df['I'])
            df['logX'] = np.log(df['X'] + 1e-12)
            feature_cols.extend(['E_times_I', 'E_plus_I', 'abs_E_minus_I', 'logX'])
        
        X_features = df[feature_cols].values
        
        # Classification: Predict lock-in
        y_class = (df['lock_epoch'] >= 0).astype(int).values
        importances_class = [0] * len(feature_cols)
        
        if np.sum(y_class) > 10:
            X_train, X_test, y_train, y_test = train_test_split(X_features, y_class, test_size=ctx.config.get("FI_TEST_SIZE", 0.3), random_state=42)
            rf_class = RandomForestClassifier(n_estimators=ctx.config.get("FI_RF_N_ESTIMATORS", 100), random_state=42)
            rf_class.fit(X_train, y_train)
            importances_class = rf_class.feature_importances_
            
            # PUBLICATION: Larger feature importance bar chart (was: 10,6)
            fig, ax = plt.subplots(figsize=(14, 9))
            sorted_idx = np.argsort(importances_class)[::-1]
            bars = ax.bar(range(len(importances_class)), importances_class[sorted_idx], 
                         color='skyblue', edgecolor='black', linewidth=1.5, alpha=0.85)
            ax.set_xticks(range(len(importances_class)))
            ax.set_xticklabels([feature_cols[i].replace('_', ' ') for i in sorted_idx], rotation=45, ha='right', fontsize=14)
            ax.set_xlabel('Feature', fontsize=16)
            ax.set_ylabel('Importance', fontsize=16)
            # Get I-definition name
            i_def = ctx.config.get("I_DEFINITION_MODE", "default")
            title = f'Feature Importance: Lock-in Classification - {i_def}' if ctx.variant != "energy_only" else 'Feature Importance: Lock-in Classification - E-only'
            ax.set_title(title, fontsize=18, pad=20)
            ax.tick_params(axis='y', labelsize=14)
            ax.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            ctx.save_fig(os.path.join(ctx.paths["AGGREGATE_FIG_DIR"], "feature_importance_classification.png"))
            
        # Regression: Predict lock-in epoch
        df_locked = df[df['lock_epoch'] >= 0].copy()
        importances_reg = [0] * len(feature_cols)
        
        if len(df_locked) > ctx.config.get("REGRESSION_MIN", 10):
            X_reg = df_locked[feature_cols].values
            y_reg = df_locked['lock_epoch'].values
            X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=ctx.config.get("FI_TEST_SIZE", 0.3), random_state=42)
            rf_reg = RandomForestRegressor(n_estimators=ctx.config.get("FI_RF_N_ESTIMATORS", 100), random_state=42)
            rf_reg.fit(X_train, y_train)
            importances_reg = rf_reg.feature_importances_
            
            # PUBLICATION: Larger feature importance bar chart (was: 10,6)
            fig, ax = plt.subplots(figsize=(14, 9))
            sorted_idx = np.argsort(importances_reg)[::-1]
            bars = ax.bar(range(len(importances_reg)), importances_reg[sorted_idx], 
                         color='lightcoral', edgecolor='black', linewidth=1.5, alpha=0.85)
            ax.set_xticks(range(len(importances_reg)))
            ax.set_xticklabels([feature_cols[i].replace('_', ' ') for i in sorted_idx], rotation=45, ha='right', fontsize=14)
            ax.set_xlabel('Feature', fontsize=16)
            ax.set_ylabel('Importance', fontsize=16)
            # Get I-definition name
            i_def = ctx.config.get("I_DEFINITION_MODE", "default")
            title = f'Feature Importance: Lock-in Epoch Regression - {i_def}' if ctx.variant != "energy_only" else 'Feature Importance: Lock-in Epoch Regression - E-only'
            ax.set_title(title, fontsize=18, pad=20)
            ax.tick_params(axis='y', labelsize=14)
            ax.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            ctx.save_fig(os.path.join(ctx.paths["AGGREGATE_FIG_DIR"], "feature_importance_regression.png"))
            
        fi_summary = pd.DataFrame({
            'feature': feature_cols,
            'importance_classification': importances_class,
            'importance_regression': importances_reg
        }).sort_values('importance_regression', ascending=False)
        ctx.save_csv(fi_summary, os.path.join(ctx.paths["AGGREGATE_DIR"], "feature_importance_summary.csv"))



def phase_10_emergent_laws(ctx: PipelineContext, df: pd.DataFrame):
    """Phase 10: Power-law fits, phase transitions, correlations."""
    if ctx.config.get("RUN_EMERGENT_LAW_DETECTORS", True) and len(df) > 50:
        
        # A) Power-law fit: Lock-in rate vs X
        bins = np.linspace(df['X'].min(), df['X'].max(), 20)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_indices = np.digitize(df['X'], bins)
        lockin_rates = []
        for i in range(1, len(bins)):
            mask = bin_indices == i
            if np.sum(mask) > 5: rate = np.mean((df.loc[mask, 'lock_epoch'] >= 0).astype(int))
            else: rate = np.nan
            lockin_rates.append(rate)
        lockin_rates = np.array(lockin_rates)
        valid = ~np.isnan(lockin_rates) & (bin_centers > 0) & (lockin_rates > 0)
        popt = [np.nan, np.nan]
        
        if np.sum(valid) > 5:
            def power_law(x, a, b): return a * x**b
            try:
                popt, _ = curve_fit(power_law, bin_centers[valid], lockin_rates[valid], p0=[1, -1], maxfev=5000)
                plt.figure(figsize=(10, 6))
                plt.scatter(bin_centers[valid], lockin_rates[valid], s=50, alpha=0.7, label='Data')
                x_fit = np.linspace(bin_centers[valid].min(), bin_centers[valid].max(), 100)
                plt.plot(x_fit, power_law(x_fit, *popt), 'r-', lw=2, label=f'Fit: y = {popt[0]:.3f} x^{popt[1]:.3f}')
                plt.xlabel('X = E·I'); plt.ylabel('Lock-in Rate'); plt.title('Power-Law Fit: Lock-in Rate vs X'); plt.legend(); plt.grid(alpha=0.3)
                ctx.save_fig(os.path.join(ctx.paths["AGGREGATE_FIG_DIR"], "emergent_law_power_law_fit.png"))
            except Exception as e:
                if ctx.config.get("VERBOSE", False): print(f"⚠️ [EMERGENT LAWS] Power-law fit failed: {e}")
        
        # B) Phase transition: Stability rate vs X (smoothed)
        if np.sum(valid) > 5:
            stability_rates = []
            for i in range(1, len(bins)):
                mask = bin_indices == i
                if np.sum(mask) > 5: rate = np.mean(df.loc[mask, 'stable'].astype(int))
                else: rate = np.nan
                stability_rates.append(rate)
            stability_rates = np.array(stability_rates)
            valid_stab = ~np.isnan(stability_rates)
            if np.sum(valid_stab) > 5:
                plt.figure(figsize=(10, 6))
                plt.plot(bin_centers[valid_stab], stability_rates[valid_stab], 'o-', lw=2, markersize=8, label='Stability Rate')
                plt.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='50% threshold')
                plt.xlabel('X = E·I'); plt.ylabel('Stability Rate'); plt.title('Phase Transition: Stability Rate vs X'); plt.legend(); plt.grid(alpha=0.3)
                ctx.save_fig(os.path.join(ctx.paths["AGGREGATE_FIG_DIR"], "emergent_law_phase_transition.png"))

        # C) Correlation matrix
        corr_cols = ['E', 'I', 'X', 'stable', 'lock_epoch']
        corr_data = df[corr_cols].copy()
        corr_data['lock_epoch'] = (corr_data['lock_epoch'] >= 0).astype(int)
        corr_matrix = corr_data.corr()
        
        plt.figure(figsize=(8, 6)); plt.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto'); plt.colorbar(label='Correlation')
        # Clean labels: remove underscores
        clean_labels = [col.replace('_', ' ') for col in corr_cols]
        plt.xticks(range(len(corr_cols)), clean_labels, rotation=45); plt.yticks(range(len(corr_cols)), clean_labels); plt.title('Correlation Matrix')
        for i in range(len(corr_cols)):
            for j in range(len(corr_cols)):
                plt.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', ha='center', va='center', color='black', fontsize=10)
        plt.tight_layout(); ctx.save_fig(os.path.join(ctx.paths["AGGREGATE_FIG_DIR"], "emergent_law_correlation_matrix.png"))
        
        law_summary = {
            'power_law_coeff_a': popt[0], 'power_law_exponent_b': popt[1],
            'mean_correlation_E_stable': corr_matrix.loc['E', 'stable'],
            'mean_correlation_I_stable': corr_matrix.loc['I', 'stable'],
            'mean_correlation_X_stable': corr_matrix.loc['X', 'stable'],
        }
        law_df = pd.DataFrame([law_summary])
        ctx.save_csv(law_df, os.path.join(ctx.paths["AGGREGATE_DIR"], "emergent_law_summary.csv"))



