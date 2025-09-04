config.py
# ===================================================================================
# MASTER CONTROLLER for TQE universe simulation of Energy–Information (E,I) dynamics
# ===================================================================================
# Author: Stefan Len
# ===================================================================================

from copy import deepcopy
import os

# ---------------------------
# Top-level, human-readable
# ---------------------------
MASTER_CTRL = {
    "META": {
        # Run labeling
        "RUN_ID_PREFIX": "TQE_(E,I)_UNIVERSE_SIMULATION",
        "RUN_ID_FORMAT": "%Y%m%d_%H%M%S",
        "CODE_VERSION": "2025-09-03a",
        "DESCRIPTION": "Monte Carlo universes with Energy–Information coupling; KL × Shannon configurable.",
        # Append EI/E tag at the end of run_id for clarity (e.g. ..._20250903_123000-EI)
        "append_ei_to_run_id": True,
    },

    # ---------------------------
    # Pipeline switches (high-level)
    # ---------------------------
    "PIPELINE": {
        # Global channel toggle
        "use_information": True,            # False => E-only baseline

        # Early stages (t < 0)
        "run_energy_sampling": True,        # E0 sampling
        "run_info_bootstrap": True,         # I0 (KL+Shannon) seeding (only if use_information=True)
        "run_fluctuation": True,            # fluctuation with E×I coupling from the start
        "run_superposition": True,          # optional quantum/superposed pass

        # Later stages
        "run_lockin": True,                 # law lock-in detection
        "run_expansion": True,              # expansion dynamics
        "run_anomaly_scan": True,           # CMB/sky-map anomaly pass
        "run_xai": True,                    # SHAP/LIME
    },

    # ---------------------------
    # Energy sampling (E0)
    # ---------------------------
    "ENERGY": {
        "distribution": "lognormal",
        "log_mu": 2.5,
        "log_sigma": 0.8,
        "trunc_low": None,
        "trunc_high": None,
        "num_universes": 5000,              # Monte Carlo population
        "time_steps": 800,                  # generic steps for simple stability loops
        "lockin_epochs": 500,               # epochs for law lock-in dynamics
        "expansion_epochs": 800,            # epochs for expansion dynamics
        "seed": None,                       # None => auto-generate
    },
    "ENERGY_SAMPLING": {
        "save_raw_arrays": True,            # save E0 vector(s)
        "plot_histogram": True,             # E0 histogram PNG
        "hist_bins": 60,
    },

    # ---------------------------
    # Information bootstrap (I0 via KL & Shannon)
    # ---------------------------
    "INFORMATION": {
        # Component toggles
        "use_kl": True,
        "use_shannon": True,

        # Hilbert space + numerics
        "hilbert_dim": 8,
        "kl_eps": 1e-12,

        # Fusion rule of components to scalar I in [0,1]
        #   "product"  => I = I_kl * I_shannon
        #   "weighted" => I = w_kl*I_kl + w_sh*I_shannon  (weights normalized)
        "fusion": "product",
        "weight_kl": 0.5,
        "weight_shannon": 0.5,

        # Post-processing
        "exponent": 1.0,                    # I <- I ** exponent
        "floor_eps": 0.0,                   # clamp minimum (avoid exact zeros if needed)
    },
    "INFORMATION_BOOTSTRAP": {
        "seed_from_energy": True,           # correlate RNG stream with ENERGY.seed if set
        "add_noise_eps": 1e-9,              # numerical floor for normalizations
        "save_raw_arrays": True,            # save I_kl, I_sh, and fused I0
        "plot_components": True,            # plot distributions and KL vs Shannon
    },

    # ---------------------------
    # Coupling X ≡ f(E, I)
    # ---------------------------
    "COUPLING_X": {
        # mode:
        #   "product"       => X = E * (alpha_I * I)
        #   "E_plus_I"      => X = (E + alpha_I * I)
        #   "E_times_I_pow" => X = E * (alpha_I * I) ** I_power
        "mode": "product",
        "alpha_I": 0.8,
        "I_power": 1.0,
        "scale": 1.0,
    },

    # ---------------------------
    # Fluctuation (t<0 dynamics with E×I from the start)
    # ---------------------------
    "FLUCTUATION": {
        "steps": 250,
        "dt": 1.0,
        "sigma_scale": 1.0,                 # multiplies NOISE.exp_noise_base
        "drift_mode": "none",               # "none" | "ou" | "custom"
        "save_trajectories": True,          # CSV/NPY for E(t), I(t), X(t)
        "plot_timeseries": True,            # PNG: sample time-series
        "plot_phase": True,                 # PNG: E vs I / X
    },

    # ---------------------------
    # Superposition / quantum pass (optional)
    # ---------------------------
    "SUPERPOSITION": {
        "enabled": True,
        "use_qutip": True,                  # graceful fallback if QuTiP not installed
        "n_realizations": 64,               # Monte Carlo realizations
        "save_states": False,               # potentially large; keep off by default
        "plot_observables": True,           # PNG plots of observables / norms / overlaps
    },

    # ---------------------------
    # Stability / lock-in thresholds
    # ---------------------------
    "STABILITY": {
        # Relative calmness thresholds (per-step or rolling)
        "rel_eps_stable": 0.010,
        "rel_eps_lockin": 5e-3,

        # Consecutive calm steps required
        "calm_steps_stable": 10,
        "calm_steps_lockin": 12,

        # Lock-in gating rules
        "min_lockin_epoch": 200,
        "lockin_window": 10,                # rolling window size
        "lockin_roll_metric": "median",     # "mean" | "median" | "max"
        "lockin_requires_stable": True,     # must be stable before lock-in
        "lockin_min_stable_epoch": 0,       # extra delay after stable_at
    },

    # ---------------------------
    # Goldilocks zone over X
    # ---------------------------
    "GOLDILOCKS": {
        # Mode:
        #   "heuristic" => fixed window from center/width
        #   "dynamic"   => estimate from empirical stability curve
        "mode": "dynamic",

        # Heuristic params (used if mode == "heuristic")
        "E_center": 4.0,
        "E_width": 4.0,

        # Dynamic extraction settings
        "threshold_frac_of_peak": 0.85,     # retain xs where P(stable) >= 0.85 * peak
        "fallback_margin": 0.10,            # ±10% around peak if curve is flat
        "sigma_alpha": 1.5,                 # curvature strength inside window
        "outside_penalty": 5,               # noise multiplier outside window

        # Stability curve binning/smoothing
        "stab_bins": 40,
        "spline_k": 3,                      # cubic if enough points
        "stab_min_count": 10,               # min samples per bin
    },

    # ---------------------------
    # Noise model (used by lock-in loop and fluctuation)
    # ---------------------------
    "NOISE": {
        "exp_noise_base": 0.12,             # baseline sigma0
        "ll_base_noise": 8e-4,              # absolute floor
        "decay_tau": 500,                   # e-folding time
        "floor_frac": 0.25,                 # portion of initial sigma preserved
        "coeff_A": 1.0,                     # per-var multipliers
        "coeff_ns": 0.10,
        "coeff_H": 0.20,
    },

    # ---------------------------
    # BEST_UNIVERSE
    # ---------------------------
   
    "run_best_universe": True,     # turn on/off the Best Universe stage

    # --- add new section in MASTER_CTRL ---
    "BEST_UNIVERSE": {
        # How many top items to render as individual PNGs (set 0 to disable)
        "top_k_png": 1,

        # Scoring weights (linear score; higher is better)
        # score = w_growth*z(S_final) + w_speed*z(-lockin_at_pos) + w_stability*stable_flag
        "weights": {
            "growth": 1.0,      # favors large S_final
            "speed":  0.7,      # favors earlier lock-in (smaller lockin_at)
            "stability": 0.3    # favors universes that reached stability
        },

        # Normalization safety
        "eps": 1e-9,

        # Which columns are expected/used (only change if átnevezted a korábbi modulok kimeneteit)
        "columns": {
            "id": "universe_id",
            "s_final": "S_final",
            "lockin": "lockin_at",
            "stable_flag": "stable"   # 0/1
        },

        # Figure style
        "plot": {
            "dpi": 180,
            "curve_color": None,     # default Matplotlib color if None
            "annot_color": "red",
        }
    }

    # ---------------------------
    # Finetune diag
    # ---------------------------
    "FINETUNE_DIAG": {
        "top_k": 1,
        "targets": {
            "rms":   {"target": 1.0, "tol": 0.25, "weight": 1.0},
            "alpha": {"target": 2.9, "tol": 0.6,  "weight": 1.0},
            "corr_len": {"min": 2.0, "max": 40.0, "tol": 2.0, "weight": 0.7},
            "skew":  {"target": 0.0, "tol": 0.15, "weight": 0.5},
            "kurt":  {"target": 0.0, "tol": 0.3,  "weight": 0.5},
        }
    }

    # ---------------------------
    # Expansion dynamics (optional)
    # ---------------------------
    "EXPANSION": {
        "growth_base": 1.005,               # reuse NOISE.exp_noise_base as amplitude if needed
    },

    # ---------------------------
    # CMB / anomaly detection parameters
    # ---------------------------
    "ANOMALY": {
        "enabled": True,
        "map": {
            "resolution_nside": 128,        # or 256/512 if you use HEALPix
            "beam_fwhm_deg": 1.0,           # smoothing for map (if applicable)
            "seed_per_map": True,           # reproducible per-universe maps
        },
        "targets": [
            {"name": "cold_spot", "enabled": True, "patch_deg": 10.0, "zscore_thresh": 3.0},
            {"name": "hemispheric_asymmetry", "enabled": True, "l_max": 40, "pval_thresh": 0.05},
            {"name": "quad_oct_align", "enabled": False, "l2l3_align_deg": 20.0},
        ],
        "save_cutouts": True,
        "save_metrics_csv": True,
    },

    # ---------------------------
    # XAI stack
    # ---------------------------
    "XAI": {
        "run_shap": True,
        "run_lime": True,
        "lime_num_features": 5,
        "test_size": 0.25,
        "test_random_state": 42,
        "rf_n_estimators": 400,
        "rf_class_weight": None,
        "sklearn_n_jobs": -1,
        "regression_min": 10,
        "max_shap_samples": 1000,
        "shap_background_size": 200,
    },

    # ---------------------------
    # Outputs / IO
    # ---------------------------
    "ENV": {
        "auto_detect": True,                # detect Colab / Desktop / Cloud
        "force_environment": None,          # force environment ("colab", "desktop", "cloud")
        "colab_markers": ["COLAB_RELEASE_TAG", "COLAB_BACKEND_VERSION"],
    },

    "OUTPUTS": {
        "save_figs": True,                  # save generated figures
        "save_json": True,                  # save JSON summaries
        "save_csv": True,                   # save CSV outputs
        "max_figs_to_save": None,           # None => no cap

        # Save-per-stage fine switches
        "save_per_stage": {
            "energy_sampling": True,
            "information_bootstrap": True,
            "fluctuation": True,
            "superposition": True,
            "lockin": True,
            "expansion": True,
            "anomaly": True,
            "xai": True,
        },

        # File name tagging
        "tag_ei_in_filenames": True,       # append _EI or _E to filenames
        "tag_profile_in_runid": True,      # include profile into run_id (e.g. -paper)

        # Local file system outputs
        "local": {
            "base_dir": "./",
            "fig_subdir": "figs",
            "allow_exts": [".png", ".fits", ".csv", ".json", ".txt", ".npy"],
            "prefer_desktop": True,
            "desktop_subdir": "TQE_Output",
            "desktop_env_var": "TQE_DESKTOP_DIR",
        },

        # Google Colab Drive outputs
        "colab_drive": {
            "enabled": True,
            "base_dir": "/content/drive/MyDrive/TQE_(E,I)_KL_Shannon",
        },

        # Cloud bucket outputs (e.g. Google Cloud Storage, AWS S3)
        "cloud": {
            "enabled": False,
            "bucket_url": None,             # e.g. "gs://my-bucket/tqe/"
        },

        # Mirroring settings (allow saving to multiple targets at once)
        "mirroring": {
            "enabled": True,
            "targets": ["local", "colab_drive"],  # add "cloud" if needed
        },

        # Plot toggles (global defaults)
        "plot_avg_lockin": True,
        "plot_lockin_hist": True,
        "plot_stability_basic": False,
        "verbose": True,
    },

    # ---------------------------
    # Debug / sanity checks
    # ---------------------------
    "DEBUG": {
        "assert_non_nan": True,
        "preview_first_n": 5,
    },

    # ---------------------------
    # Reproducibility
    # ---------------------------
    "REPRO": {
        "use_strict_seed": True,            # set env vars to limit threads, etc.
        "per_universe_seed_mode": "rng",    # "rng" | "np_random"
        "env_thread_caps": {
            "PYTHONHASHSEED": "0",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        },
    },

    # ---------------------------
    # Runtime knobs (plots, logging)
    # ---------------------------
    "RUNTIME": {
        "matplotlib_dpi": 180,
        "log_level": "INFO",
    },

    # ---------------------------
    # Profiles (named overrides)
    # ---------------------------
    "PROFILES": {
        # Lightweight smoke test / Colab demo
        "demo": {
            "ENERGY": {"num_universes": 800, "time_steps": 200, "lockin_epochs": 200, "expansion_epochs": 200},
            "ANOMALY": {"map": {"resolution_nside": 64}},
        },

        # Publication-style settings (balanced)
        "paper": {
            "ENERGY": {"num_universes": 5000, "time_steps": 800, "lockin_epochs": 500, "expansion_epochs": 800},
            "ANOMALY": {"map": {"resolution_nside": 128}},
            "XAI": {"rf_n_estimators": 400},
        },

        # Bigger cloud run
        "full_cloud": {
            "ENERGY": {"num_universes": 20000},
            "ANOMALY": {"map": {"resolution_nside": 256}},
            "OUTPUTS": {"cloud": {"enabled": True, "bucket_url": "gs://YOUR_BUCKET/tqe_runs/"}},
        },

        # Force E×I with both components on
        "ei_only": {
            "PIPELINE": {"use_information": True},
            "INFORMATION": {"use_kl": True, "use_shannon": True, "fusion": "product"},
        },

        # E-only baseline (no info channel)
        "e_only": {
            "PIPELINE": {"use_information": False},
        },
    },
}

# ======================================================
# Small helper: deep-merge dict (right overrides left)
# ======================================================
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

# ======================================================
# Select profile here (or from CLI) and build ACTIVE cfg
# ======================================================
SELECTED_PROFILE = os.environ.get("TQE_PROFILE", "paper")  # "demo" | "paper" | "full_cloud" | "ei_only" | "e_only"

def resolve_profile(profile_name: str):
    base = deepcopy(MASTER_CTRL)
    profs = base.pop("PROFILES", {})
    chosen = profs.get(profile_name, {})
    active = _deep_merge(base, chosen)

    # If strict seed requested, expose env caps (runner can apply)
    if active["REPRO"]["use_strict_seed"]:
        for k, v in active["REPRO"]["env_thread_caps"].items():
            os.environ[k] = str(v)

    # Optionally tag the run_id with the selected profile and EI/E mode
    if base["OUTPUTS"].get("tag_profile_in_runid", False):
        os.environ["TQE_PROFILE_TAG"] = profile_name

    return active

ACTIVE = resolve_profile(SELECTED_PROFILE)
