# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Stefan Len
#
# Bayesian analysis module
#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ..config.master_ctrl import MASTER_CTRL
from ..core.pipeline_context import PipelineContext

def compute_bayesian_model_selection(ctx: PipelineContext, df: pd.DataFrame, planck_chi2: float) -> dict:
    """
    Compute Bayesian Information Criterion (BIC), Akaike Information Criterion (AIC),
    and prepare for Bayes Factor calculation.
    
    Args:
        ctx: Pipeline context
        df: DataFrame with universe results
        planck_chi2: Chi-squared from Planck validation
    
    Returns:
        dict: Bayesian metrics (BIC, AIC, log_likelihood, n_params, n_data)
    """
    if not ctx.config.get("ENABLE_BAYESIAN_ANALYSIS", False):
        return {}
    
    # Number of data points (CMB pixels + Planck observables)
    n_cmb_pixels = ctx.config.get("CMB_NPIX", 64**2)
    n_planck_obs = 6  # (H0, Omega_m, Omega_Lambda, sigma8, n_s, tau)
    n_data = n_cmb_pixels + n_planck_obs
    
    # Number of free parameters (X_SCALE, ALPHA_I, + model-specific)
    k = 2  # X_SCALE, ALPHA_I
    if ctx.variant != "energy_only":
        k += 1  # I-definition adds complexity
    
    # Log-likelihood from chi-squared
    # L = exp(-χ²/2) → log(L) = -χ²/2
    log_likelihood = -0.5 * planck_chi2
    
    # Bayesian Information Criterion (BIC)
    # BIC = k*log(n) - 2*log(L) = k*log(n) + χ²
    bic = k * np.log(n_data) + planck_chi2
    
    # Akaike Information Criterion (AIC)
    # AIC = 2*k - 2*log(L) = 2*k + χ²
    aic = 2 * k + planck_chi2
    
    # Corrected AIC (AICc) for small sample size
    # AICc = AIC + 2k(k+1)/(n-k-1)
    if n_data > k + 1:
        aicc = aic + (2 * k * (k + 1)) / (n_data - k - 1)
    else:
        aicc = np.inf
    
    # Store results
    bayesian_metrics = {
        "BIC": float(bic),
        "AIC": float(aic),
        "AICc": float(aicc),
        "log_likelihood": float(log_likelihood),
        "n_parameters": int(k),
        "n_data_points": int(n_data),
        "chi_squared": float(planck_chi2),
        "chi_squared_reduced": float(planck_chi2 / (n_data - k))
    }
    
    return bayesian_metrics



def run_nested_sampling(ctx: PipelineContext, df: pd.DataFrame) -> dict:
    """
    Run nested sampling to compute Bayesian evidence for model comparison.
    Uses dynesty library for nested sampling.
    
    Args:
        ctx: Pipeline context
        df: DataFrame with universe results
    
    Returns:
        dict: Nested sampling results (log_evidence, evidence_error, samples, weights)
    """
    if not ctx.config.get("ENABLE_NESTED_SAMPLING", False) or not DYNESTY_AVAILABLE:
        return {}
    
    if ctx.config.get("VERBOSE", True):
        print("\n[NESTED SAMPLING] Starting Bayesian evidence calculation...")
    
    # Define log-likelihood function
    def log_likelihood_func(theta):
        """
        Log-likelihood for given parameters theta = [X_SCALE, ALPHA_I].
        Based on Planck chi-squared and CMB anomaly matches.
        """
        X_SCALE, ALPHA_I = theta
        
        # Simulate small Monte Carlo with these parameters
        rng_local = np.random.default_rng(ctx.master_seed + 999)
        physics_tmp = PhysicsEngine(ctx.config.copy(), rng_local)
        
        # Sample E-I pairs
        n_samples = 50  # Small for speed
        E_samples = rng_local.lognormal(mean=ctx.config.get("E_LOG_MU", 0.5), 
                                         sigma=ctx.config.get("E_LOG_SIGMA", 0.8), 
                                         size=n_samples)
        
        # Compute chi-squared proxy
        chi2_proxy = 0.0
        for E in E_samples:
            if ctx.variant == "energy_only":
                I = 0.0
                X = E * X_SCALE  # E-only mode
            else:
                I_defs = physics_tmp.compute_all_I_definitions(E, a=1.0)
                I = I_defs.get(ctx.config.get("I_DEFINITION_MODE", "kl_shannon"), 0.5)
            
                # FIXED: Use compute_coupling to respect X_MODE!
                X = physics_tmp.compute_coupling(E, I)
            
            # Planck reference: E (Omega_Lambda) = 0.7
            delta_E = abs(E - 0.7)
            chi2_proxy += delta_E**2
        
        chi2_proxy /= n_samples
        
        # Log-likelihood
        log_like = -0.5 * chi2_proxy * 100  # Scale factor
        
        return log_like
    
    # Define prior transform (uniform priors)
    def prior_transform(u):
        """Transform unit cube [0,1]^2 to parameter space."""
        x_min, x_max = ctx.config.get("BAYESIAN_PRIOR_X_SCALE", (10.0, 50.0))
        a_min, a_max = ctx.config.get("BAYESIAN_PRIOR_ALPHA_I", (0.1, 2.0))
        
        X_SCALE = x_min + (x_max - x_min) * u[0]
        ALPHA_I = a_min + (a_max - a_min) * u[1]
        
        return np.array([X_SCALE, ALPHA_I])
    
    # Run nested sampling
    try:
        sampler = dynesty.NestedSampler(
            log_likelihood_func, 
            prior_transform, 
            ndim=2,
            nlive=ctx.config.get("NESTED_SAMPLING_NLIVE", 500),
            bound='multi',
            sample='auto'
        )
        
        sampler.run_nested(
            dlogz=ctx.config.get("NESTED_SAMPLING_DLOGZ", 0.5),
            maxiter=ctx.config.get("NESTED_SAMPLING_MAX_ITER", 10000),
            print_progress=ctx.config.get("VERBOSE", True)
        )
        
        results = sampler.results
        
        # Extract key results
        # FIX: Handle numpy arrays that may contain scalars (use np.atleast_1d to ensure array)
        # Critical: importance_weights() can return a scalar, must use np.atleast_1d!
        importance_wts = results.importance_weights()
        
        # FIX: Safely extract scalar values from arrays
        # Critical: logz/logzerr can be 0-d, 1-d, or even nested arrays!
        # Always flatten first, then extract last element
        logz_flat = np.atleast_1d(results.logz).flatten()
        logzerr_flat = np.atleast_1d(results.logzerr).flatten()
        
        # Extract last element (guaranteed to be scalar after flatten)
        log_evidence_val = float(logz_flat[-1]) if len(logz_flat) > 0 else 0.0
        log_evidence_err = float(logzerr_flat[-1]) if len(logzerr_flat) > 0 else 0.0
        
        # FIX: Safely convert all dynesty results to lists (handle scalars AND arrays)
        # Critical: ALL dynesty results can be scalars, 0-d, 1-d, or nested arrays!
        # Strategy: flatten everything first, then convert
        nested_results = {
            "log_evidence": float(log_evidence_val),
            "log_evidence_error": float(log_evidence_err),
            "n_iterations": int(np.atleast_1d(results.niter).flatten()[0]),
            "n_calls": int(np.atleast_1d(results.ncall).flatten()[0]),
            "samples": np.atleast_2d(results.samples).tolist() if hasattr(results.samples, 'shape') else [],
            "weights": np.atleast_1d(importance_wts).flatten().tolist(),
            "logwt": np.atleast_1d(results.logwt).flatten().tolist(),
            "logl": np.atleast_1d(results.logl).flatten().tolist()
        }
        
        if ctx.config.get("VERBOSE", True):
            print(f"[NESTED SAMPLING] log(Evidence) = {nested_results['log_evidence']:.2f} ± {nested_results['log_evidence_error']:.2f}")
            print(f"[NESTED SAMPLING] Completed in {nested_results['n_iterations']} iterations ({nested_results['n_calls']} likelihood calls)")
        
        # Save samples to CSV (ensure 2D array for DataFrame, flatten all 1D arrays)
        samples_2d = np.atleast_2d(results.samples)
        samples_df = pd.DataFrame(samples_2d, columns=["X_SCALE", "ALPHA_I"])
        samples_df["weight"] = np.atleast_1d(importance_wts).flatten()  # Flatten to 1D
        samples_df["log_likelihood"] = np.atleast_1d(results.logl).flatten()  # Flatten to 1D
        ctx.save_csv(samples_df, os.path.join(ctx.paths["AGGREGATE_DIR"], "nested_sampling_samples.csv"))
        
        # Generate corner plot if enabled
        if ctx.config.get("ENABLE_CORNER_PLOTS", False) and CORNER_AVAILABLE:
            generate_corner_plot(ctx, np.atleast_2d(results.samples), np.atleast_1d(importance_wts).flatten())
        
        return nested_results
        
    except Exception as e:
        if ctx.config.get("VERBOSE", True):
            print(f"[NESTED SAMPLING][ERROR] Failed: {e}")
        return {}



def generate_corner_plot(ctx: PipelineContext, samples: np.ndarray, weights: np.ndarray):
    """
    Generate corner plot showing parameter posterior distributions.
    
    Args:
        ctx: Pipeline context
        samples: (N, ndim) array of samples
        weights: (N,) array of sample weights
    """
    if not CORNER_AVAILABLE:
        return
    
    try:
        # Get optimal parameters from Goldilocks
        x_scale_opt = ctx.config.get("X_SCALE", 20.0)
        alpha_i_opt = ctx.config.get("ALPHA_I", 0.9)
        
        # Create corner plot
        fig = corner.corner(
            samples,
            weights=weights,
            labels=["$X_{\\rm SCALE}$", "$\\alpha_I$"],
            truths=[x_scale_opt, alpha_i_opt],
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_kwargs={"fontsize": 10, "pad": 15},  # Smaller font, more padding
            title_fmt=".2f",  # Shorter number format
            color='#1f77b4',
            truth_color='red',
            hist_kwargs={'density': True, 'alpha': 0.6},
            contour_kwargs={'colors': '#1f77b4', 'linewidths': 1.5}
        )
        
        # Add title with more space
        fig.suptitle(f"Parameter Posterior Distributions\n(I-definition: {ctx.config.get('I_DEFINITION_MODE', 'default')})", 
                     fontsize=14, y=0.98)  # Lower position to avoid overlap
        
        # Adjust layout to prevent overlap
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space at top for suptitle
        
        # Save figure
        i_def = ctx.config.get("I_DEFINITION_MODE", "default")
        filename = f"corner_plot_{i_def}.png"
        ctx.save_fig(os.path.join(ctx.paths["AGGREGATE_FIG_DIR"], filename))
        
        if ctx.config.get("VERBOSE", True):
            print(f"[CORNER PLOT] Saved: {filename}")
        
    except Exception as e:
        if ctx.config.get("VERBOSE", True):
            print(f"[CORNER PLOT][ERROR] Failed: {e}")



def save_bayesian_metrics_csv(ctx: PipelineContext, bayesian_metrics: dict, nested_results: dict):
    """
    Save Bayesian metrics to CSV for easy comparison across I-definitions.
    
    Args:
        ctx: Pipeline context
        bayesian_metrics: BIC/AIC metrics dict
        nested_results: Nested sampling results dict
    """
    if not bayesian_metrics and not nested_results:
        return
    
    # Combine metrics
    combined_metrics = {
        "i_definition": ctx.config.get("I_DEFINITION_MODE", "default"),
        "variant": ctx.variant,
        "BIC": bayesian_metrics.get("BIC", np.nan),
        "AIC": bayesian_metrics.get("AIC", np.nan),
        "AICc": bayesian_metrics.get("AICc", np.nan),
        "log_likelihood": bayesian_metrics.get("log_likelihood", np.nan),
        "chi_squared": bayesian_metrics.get("chi_squared", np.nan),
        "chi_squared_reduced": bayesian_metrics.get("chi_squared_reduced", np.nan),
        "n_parameters": bayesian_metrics.get("n_parameters", np.nan),
        "n_data_points": bayesian_metrics.get("n_data_points", np.nan),
        "log_evidence": nested_results.get("log_evidence", np.nan),
        "log_evidence_error": nested_results.get("log_evidence_error", np.nan),
        "nested_n_iterations": nested_results.get("n_iterations", np.nan),
        "nested_n_calls": nested_results.get("n_calls", np.nan),
    }
    
    # Save to CSV
    df = pd.DataFrame([combined_metrics])
    i_def = ctx.config.get("I_DEFINITION_MODE", "default")
    filename = f"bayesian_metrics_{i_def}.csv"
    ctx.save_csv(df, os.path.join(ctx.paths["AGGREGATE_DIR"], filename))
    
    if ctx.config.get("VERBOSE", True):
        print(f"[BAYESIAN CSV] Saved: {filename}")



def plot_bayesian_comparison(ctx: PipelineContext, bayesian_metrics: dict):
    """
    Generate bar chart comparing BIC, AIC, and chi-squared.
    
    Args:
        ctx: Pipeline context
        bayesian_metrics: Bayesian metrics dict
    """
    if not bayesian_metrics:
        return
    
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        i_def = ctx.config.get("I_DEFINITION_MODE", "default")
        
        # Plot 1: BIC/AIC comparison
        ax1 = axes[0]
        metrics_names = ['BIC', 'AIC', 'AICc']
        metrics_values = [bayesian_metrics.get(m, 0) for m in metrics_names]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        ax1.bar(metrics_names, metrics_values, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_ylabel('Value', fontsize=13)
        ax1.set_title('Information Criteria\n(Lower = Better Model)', fontsize=13)
        ax1.grid(axis='y', alpha=0.3)
        
        # Plot 2: Chi-squared
        ax2 = axes[1]
        chi2 = bayesian_metrics.get("chi_squared", 0)
        chi2_reduced = bayesian_metrics.get("chi_squared_reduced", 0)
        ax2.bar(['χ²', 'χ²/dof'], [chi2, chi2_reduced], color=['#d62728', '#9467bd'], alpha=0.7, edgecolor='black')
        ax2.set_ylabel('Value', fontsize=13)
        ax2.set_title('Chi-Squared Fit to Planck', fontsize=13)
        ax2.grid(axis='y', alpha=0.3)
        
        # Plot 3: Model complexity
        ax3 = axes[2]
        k = bayesian_metrics.get("n_parameters", 0)
        n = bayesian_metrics.get("n_data_points", 0)
        ax3.bar(['Parameters (k)', 'Data Points (n)'], [k, n], color=['#8c564b', '#e377c2'], alpha=0.7, edgecolor='black')
        ax3.set_ylabel('Count', fontsize=13)
        ax3.set_title('Model Complexity', fontsize=13)
        ax3.grid(axis='y', alpha=0.3)
        ax3.set_yscale('log')
        
        fig.suptitle(f"Bayesian Model Selection - {i_def}", fontsize=16, y=1.02)
        plt.tight_layout()
        
        # Save figure
        filename = f"bayesian_comparison_{i_def}.png"
        ctx.save_fig(os.path.join(ctx.paths["AGGREGATE_FIG_DIR"], filename))
        
        if ctx.config.get("VERBOSE", True):
            print(f"[BAYESIAN PLOT] Saved: {filename}")
        
        plt.close(fig)
        
    except Exception as e:
        if ctx.config.get("VERBOSE", True):
            print(f"[BAYESIAN PLOT][ERROR] Failed: {e}")

