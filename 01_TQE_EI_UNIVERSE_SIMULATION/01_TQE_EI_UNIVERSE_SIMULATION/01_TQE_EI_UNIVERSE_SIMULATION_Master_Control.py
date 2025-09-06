# ===================================================================================
# 01_TQE_EI_UNIVERSE_SIMULATION_Master_Control.py
# ===================================================================================
# Author: Stefan Len
# ===================================================================================

from copy import deepcopy
import os

# ===================================================================================
# MASTER CONFIGURATION
# ===================================================================================
MASTER_CTRL = {
    # ---------------------------
    # Metadata and run labeling
    # ---------------------------
    "META": {
        "RUN_ID_PREFIX": "TQE_EI_UNIVERSE_SIMULATION",
        "RUN_ID_FORMAT": "%Y%m%d_%H%M%S",
        "CODE_VERSION": "2025-09-05",
        "DESCRIPTION": "Universe Monte Carlo with Energy–Information coupling, anomalies, XAI, manifest.",
        "append_ei_to_run_id": True,   # Append -EI or -E at the end of run_id
    },

    # ---------------------------
    # Pipeline switches (high-level)
    # ---------------------------
    "PIPELINE": {
        "use_information": True,          # False => Energy-only baseline

        # Early stages
        "run_energy_sampling": True,      # E0 sampling
        "run_info_bootstrap": True,       # I0 seeding (KL + Shannon)
        "run_fluctuation": True,          # Pre t=0 fluctuation dynamics
        "run_superposition": True,        # Optional quantum superposition stage

        # Core evolution
        "run_lockin": True,               # Law lock-in detection
        "run_expansion": True,            # Expansion dynamics

        # Post stages / analytics
        "run_montecarlo": True,           # Monte Carlo statistics
        "run_best_universe": True,        # Best-universe scoring
        "run_cmb_map": True,              # CMB map generation
        "run_anomaly_scan": True,         # Anomaly detection
        "run_finetune_diag": True,        # Fine-tuning diagnostics
        "run_xai": True,                  # SHAP/LIME explainability
        "run_manifest": True,             # Results manifest generation
    },

    # ---------------------------
    # Energy sampling
    # ---------------------------
    "ENERGY": {
        "distribution": "lognormal",
        "log_mu": 2.5,
        "log_sigma": 0.8,
        "trunc_low": None,
        "trunc_high": None,
        "num_universes": 5000,
        "time_steps": 800,
        "lockin_epochs": 500,
        "expansion_epochs": 800,
        "seed": None,                     # None => auto-generate
    },
    "ENERGY_SAMPLING": {
        "save_raw_arrays": True,          # Save raw sampled values
        "plot_histogram": True,           # Plot histogram of E0
        "hist_bins": 60,
    },

    # ---------------------------
    # Information bootstrap (KL + Shannon)
    # ---------------------------
    "INFORMATION": {
        "use_kl": True,                   # Enable KL divergence component
        "use_shannon": True,              # Enable Shannon entropy component
        "hilbert_dim": 8,
        "kl_eps": 1e-12,
        "fusion": "product",              # "product" | "weighted"
        "weight_kl": 0.5,
        "weight_shannon": 0.5,
        "exponent": 1.0,                  # Post-processing exponent
        "floor_eps": 0.0,                 # Minimum clamp
    },
    "INFORMATION_BOOTSTRAP": {
        "seed_from_energy": True,         # Correlate RNG with ENERGY.seed if set
        "add_noise_eps": 1e-9,            # Numerical floor for normalization
        "save_raw_arrays": True,          # Save KL, Shannon, fused I0
        "plot_components": True,          # Plot component distributions
    },

    # ---------------------------
    # Coupling definition: X ≡ f(E,I)
    # ---------------------------
    "COUPLING_X": {
        "mode": "product",                # "product" | "E_plus_I" | "E_times_I_pow"
        "alpha_I": 0.8,                   # Scaling for I
        "I_power": 1.0,                   # Power exponent for I
        "scale": 1.0,                     # Global scale
    },

    # ---------------------------
    # Fluctuation (t < 0 dynamics)
    # ---------------------------
    "FLUCTUATION": {
        "steps": 250,
        "dt": 1.0,
        "sigma_scale": 1.0,               # Multiplies NOISE.exp_noise_base
        "drift_mode": "none",             # "none" | "ou" | "custom"
        "save_trajectories": True,        # Save E(t), I(t), X(t)
        "plot_timeseries": True,          # Plot sample trajectories
        "plot_phase": True,               # Plot E vs I phase space
    },

    # ---------------------------
    # Superposition / quantum stage
    # ---------------------------
    "SUPERPOSITION": {
        "enabled": True,
        "use_qutip": True,                # Fallback gracefully if QuTiP not available
        "n_realizations": 64,             # Monte Carlo realizations
        "save_states": False,             # Save full states (large memory use)
        "plot_observables": True,         # Plot observables and overlaps
    },

    # ---------------------------
    # Stability and lock-in thresholds
    # ---------------------------
    "STABILITY": {
        "rel_eps_stable": 0.010,          # Stability threshold
        "rel_eps_lockin": 5e-3,           # Lock-in threshold
        "calm_steps_stable": 10,          # Required calm steps for stability
        "calm_steps_lockin": 12,          # Required calm steps for lock-in
        "min_lockin_epoch": 200,          # Earliest epoch for lock-in
        "lockin_window": 10,              # Rolling window size
        "lockin_roll_metric": "median",   # "mean" | "median" | "max"
        "lockin_requires_stable": True,   # Require stability before lock-in
        "lockin_min_stable_epoch": 0,     # Extra delay after stability
    },

    # ---------------------------
    # Goldilocks zone (dynamic stability window)
    # ---------------------------
    "GOLDILOCKS": {
        "mode": "dynamic",                # "heuristic" | "dynamic"
        "E_center": 4.0,                  # Used if mode = heuristic
        "E_width": 4.0,
        "threshold_frac_of_peak": 0.85,   # Fraction of peak stability
        "fallback_margin": 0.10,          # ±10% margin if flat curve
        "sigma_alpha": 1.5,               # Inside curve sharpness
        "outside_penalty": 5.0,           # Noise penalty outside window
        "stab_bins": 40,                  # Number of bins
        "spline_k": 3,                    # Cubic spline degree
        "stab_min_count": 10,             # Minimum samples per bin
    },

    # ---------------------------
    # Noise model
    # ---------------------------
    "NOISE": {
        "exp_noise_base": 0.12,           # Baseline noise
        "ll_base_noise": 8e-4,            # Absolute noise floor
        "decay_tau": 500,                 # Exponential decay constant
        "floor_frac": 0.25,               # Noise floor fraction
        "coeff_A": 1.0,                   # Multipliers
        "coeff_ns": 0.10,
        "coeff_H": 0.20,
    },

    # ---------------------------
    # Expansion dynamics
    # ---------------------------
    "EXPANSION": {
        "growth_base": 1.005,
        "gamma": 1.0,
    },

    # ---------------------------
    # Monte Carlo run
    # ---------------------------
    "MONTECARLO": {
        "save_csv": True,
        "save_json": True,
        "plot_distributions": True,
    },

    # ---------------------------
    # Best Universe scoring
    # ---------------------------
    "BEST_UNIVERSE": {
        "top_k_png": 1,                   # Number of top universes to plot
        "weights": {"growth": 1.0, "speed": 0.7, "stability": 0.3},
        "eps": 1e-9,                      # Numerical safety epsilon
        "columns": {
            "id": "universe_id",
            "s_final": "S_final",
            "lockin": "lockin_at",
            "stable_flag": "stable",
        },
        "plot": {"dpi": 180, "curve_color": None, "annot_color": "red"},
    },

    # ---------------------------
    # CMB map generation
    # ---------------------------
    "CMB_MAP": {
        "resolution_nside": 128,          # HEALPix resolution
        "beam_fwhm_deg": 1.0,             # Beam smoothing
        "seed_per_map": True,             # Per-universe reproducibility
        "save_png": True,                 # Save CMB maps as PNG
        "save_npy": True,                 # Save CMB maps as NumPy arrays
    },

    # ---------------------------
    # Anomaly detection
    # ---------------------------
    "ANOMALY": {
        "enabled": True,
        "save_cutouts": True,             # Save map cutouts
        "save_metrics_csv": True,         # Save anomaly metrics
        "targets": [
            {"name": "cold_spot", "enabled": True, "patch_deg": 10.0, "zscore_thresh": 3.0},
            {"name": "low_multipole_align", "enabled": True, "l2l3_align_deg": 20.0},
            {"name": "lack_large_angle_corr", "enabled": True, "theta_min_deg": 60.0, "num_boot": 500},
            {"name": "hemispheric_asymmetry", "enabled": True, "l_max": 40, "pval_thresh": 0.05},
        ],
    },

    # ---------------------------
    # Fine-tuning diagnostics
    # ---------------------------
    "FINETUNE_DIAG": {
        "top_k": 1,
        "targets": {
            "rms":      {"target": 1.0, "tol": 0.25, "weight": 1.0},
            "alpha":    {"target": 2.9, "tol": 0.6,  "weight": 1.0},
            "corr_len": {"min": 2.0, "max": 40.0, "tol": 2.0, "weight": 0.7},
            "skew":     {"target": 0.0, "tol": 0.15, "weight": 0.5},
            "kurt":     {"target": 0.0, "tol": 0.3,  "weight": 0.5},
        },
    },

    # ---------------------------
    # Explainable AI (SHAP / LIME)
    # ---------------------------
    "XAI": {
        "run_shap": True,
        "run_lime": True,
        "lime_num_features": 6,
        "test_size": 0.25,
        "test_random_state": 42,
        "rf_n_estimators": 400,
        "rf_class_weight": None,
        "sklearn_n_jobs": -1,
        "features": {
            "energy": True,
            "info": True,
            "coupling": True,
            "lockin": True,
            "expansion": True,
            "anomalies": True,
        },
        "targets": {
            "S_final_regression": True,
            "lockin_binary": True,
            "cold_spot_zscore": True,
            "lack_large_angle_C": True,
            "hemi_asymmetry_index": True,
        },
        "regression_min": 10,
        "max_shap_samples": 1000,
        "shap_background_size": 200,
    },

    # ---------------------------
    # Environment detection
    # ---------------------------
    "ENV": {
        "auto_detect": False,
        "force_environment": None,        # "colab" | "cloud" | "desktop"
        "colab_markers": ["COLAB_RELEASE_TAG", "COLAB_BACKEND_VERSION"],
    },

    # ---------------------------
    # Output / IO settings
    # ---------------------------
    "OUTPUTS": {
        "save_figs": True,
        "save_json": True,
        "save_csv": True,
        "max_figs_to_save": None,    # None = no limit
        # Always save everything per stage
        "save_per_stage": {
            "energy_sampling": True,
            "information_bootstrap": True,
            "fluctuation": True,
            "superposition": True,
            "lockin": True,
            "expansion": True,
            "montecarlo": True,
            "best_universe": True,
            "cmb_map": True,
            "anomaly": True,
            "finetune": True,
            "xai": True,
            "manifest": True,
        },

        # Always tag filenames clearly
        "tag_ei_in_filenames": True,      # Append -EI or -E to filenames
        "tag_profile_in_runid": True,     # Append profile tag to run_id
        
        # Force local save to Colab path
        "local": {
            "base_dir": "/content/TQE_Output",
            "fig_subdir": "figs",
            "allow_exts": [".png", ".fits", ".csv", ".json", ".txt", ".npy"],
            "prefer_desktop": False,
            "desktop_subdir": None,
            "desktop_env_var": None,
        },

         # Disable Drive and Cloud for now
        "colab_drive": {
            "enabled": True,
            "base_dir": "/content/drive/MyDrive/TQE_(E,I)_UNIVERSE_SIMULATION",
        },
        "cloud": {
            "enabled": False,
            "bucket_url": None,
        },

        # Disable mirroring (single target only)
        "mirroring": {
            "enabled": False,
            "targets": ["local", "colab_drive"],
        },

        # Plot options
        "plot_avg_lockin": True,
        "plot_lockin_hist": True,
        "plot_stability_basic": True,
         # Verbose logs
        "verbose": True,
    },

    # ---------------------------
    # Debug options
    # ---------------------------
    "DEBUG": {
        "assert_non_nan": True,
        "preview_first_n": 5,
    },

    # ---------------------------
    # Reproducibility
    # ---------------------------
    "REPRO": {
        "use_strict_seed": True,
        "per_universe_seed_mode": "rng",
        "env_thread_caps": {
            "PYTHONHASHSEED": "0",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        },
    },

    # ---------------------------
    # Runtime
    # ---------------------------
    "RUNTIME": {
        "matplotlib_dpi": 180,
        "log_level": "INFO",
    },

    # ---------------------------
    # Profiles
    # ---------------------------
    "PROFILES": {
        "demo": {
            "ENERGY": {"num_universes": 800, "time_steps": 200, "lockin_epochs": 200, "expansion_epochs": 200},
            "CMB_MAP": {"resolution_nside": 64},
        },
        "paper": {
            "ENERGY": {"num_universes": 5000, "time_steps": 800, "lockin_epochs": 500, "expansion_epochs": 800},
            "CMB_MAP": {"resolution_nside": 128},
            "XAI": {"rf_n_estimators": 400},
        },
        "full_cloud": {
            "ENERGY": {"num_universes": 20000},
            "CMB_MAP": {"resolution_nside": 256},
            "OUTPUTS": {"cloud": {"enabled": True, "bucket_url": "gs://YOUR_BUCKET/tqe_runs/"}},
        },
        "ei_only": {
            "PIPELINE": {"use_information": True},
            "INFORMATION": {"use_kl": True, "use_shannon": True, "fusion": "product"},
        },
        "e_only": {
            "PIPELINE": {"use_information": False},
        },
    },
}

# ===================================================================================
# Helper: deep-merge dicts
# ===================================================================================
def _deep_merge(base, override):
    if not isinstance(override, dict):
        return override
    out = deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out

# ===================================================================================
# Profile resolution
# ===================================================================================
SELECTED_PROFILE = os.environ.get("TQE_PROFILE", "demo")

def resolve_profile(profile_name: str):
    base = deepcopy(MASTER_CTRL)
    profs = base.pop("PROFILES", {})
    chosen = profs.get(profile_name, {})
    active = _deep_merge(base, chosen)

    # Enforce strict seed reproducibility
    if active["REPRO"]["use_strict_seed"]:
        for k, v in active["REPRO"]["env_thread_caps"].items():
            os.environ[k] = str(v)

    # Tag run_id with profile name if enabled
    if base["OUTPUTS"].get("tag_profile_in_runid", False):
        os.environ["TQE_PROFILE_TAG"] = profile_name

    return active

ACTIVE = resolve_profile(SELECTED_PROFILE)

# ---------------------------------------------------------------------------
# AUTO-ZIP & DOWNLOAD (Colab-friendly)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os, shutil, datetime
    from io_paths import resolve_output_paths  

    # 1) Default output directory (all results should land here)
    out_dir = "/content/TQE_Output"

    # Try to resolve more specific run directory if available
    try:
        paths = resolve_output_paths(ACTIVE)
        out_dir = paths.get("primary_run_dir", out_dir)
    except Exception as e:
        print("[WARN] resolve_output_paths() failed, fallback to /content/TQE_Output:", e)

    # Make sure the directory exists
    os.makedirs(out_dir, exist_ok=True)

    # 2) Create a timestamped archive
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_name = f"TQE_Output_{timestamp}"
    zip_path = shutil.make_archive(f"/content/{zip_name}", "zip", out_dir)
    print("📦 Archive created:", zip_path)

    # 3) Trigger automatic download in Colab
    try:
        from google.colab import files
        files.download(zip_path)
        print("⬇️ Download triggered.")
    except Exception:
        print("[INFO] Not in Colab, skipping auto-download.")
