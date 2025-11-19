# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Stefan Len
#
# Configuration module for TQE Analysis Pipeline

import os

# ==========================================================================================
# MASTER CONTROL PANEL
# ==========================================================================================
# Configure analysis target, ranking weights, visualization settings, and statistical options.
# Modify these parameters to customize the analysis behavior.
# ==========================================================================================
MASTER_CTRL = {
    
    # === TARGET SELECTION ===
    "TARGET_MODE": "batch_all",
    
    # TARGET_TIMESTAMP: Specify which simulation batch to analyze
    #   - None: Auto-detect latest batch (recommended for most recent results)
    #   - "YYYYMMDD_HHMMSS": Specific timestamp (e.g., "20251118_101437")
    #   
    #   Current: None - auto-detect latest batch_all run
    #   Latest batch: "20251118_101437" - batch_all with 11 runs (1 E-only + 10 E+I)
    "TARGET_TIMESTAMP": None,  # Auto-detect latest batch
    
    # === PATH CONFIGURATION ===
    # AUTO-DETECT: These are set automatically, override only if needed
    "SIMULATION_ROOT": None,          # Auto-detected (local)
    "ANALYSIS_OUTPUT_ROOT": None,     # Auto-detected (local)
    
    # === RANKING WEIGHTS (Model Selection) ===
    # THREE RANKING SYSTEMS:
    
    # RANKING SYSTEM 1: STABILITY-FOCUSED (Traditional)
    "RANKING_WEIGHTS_STABILITY": {
        "stability_rate": 0.30,           # 30% - Stable universe %
        "lockin_rate": 0.20,              # 20% - Law emergence frequency
        "planck_chi2_fit": 0.20,          # 20% - Observational validation
        "goldilocks_precision": 0.15,     # 15% - Peak uncertainty
        "cmb_anomaly_match": 0.10,        # 10% - Anomaly detection
        "bayesian_efficiency": 0.05,      # 5% - GP performance
    },
    
    # RANKING SYSTEM 2: COMPLEXITY-FOCUSED (TQE-Consistent)
    "RANKING_WEIGHTS_COMPLEXITY": {
        "complexity_score": 0.35,         # 35% - Structural complexity
        "life_compatibility": 0.25,       # 25% - Life-readiness indicators
        "information_richness": 0.20,     # 20% - I-parameter effectiveness
        "stability_quality": 0.10,        # 10% - Stability (not quantity!)
        "observational_match": 0.10,      # 10% - Planck fit
    },
    
    # RANKING SYSTEM 3: PHYSICAL-LAWS-FOCUSED
    "RANKING_WEIGHTS_PHYSICAL_LAWS": {
        "emergent_laws_quality": 0.30,    # 30% - Power-law, phase transitions
        "friedmann_consistency": 0.25,    # 25% - age, H0, Omegas vs Planck 2018
        "cmb_anomaly_match": 0.20,        # 20% - Cold spots, Axis of Evil
        "lockin_efficiency": 0.15,        # 15% - Fast, decisive law formation
        "quantum_field_realism": 0.10,    # 10% - Vacuum energy, fluctuations
    },
    
    # Active ranking system
    "RANKING_MODE": "triple",  # "stability" | "complexity" | "physical_laws" | "triple"
    
    # === VISUALIZATION SETTINGS ===
    "FIGURE_DPI": 300,                    # Plot resolution (300 for publication quality)
    "FIGURE_FORMAT": "png",               # Output format
    "PLOT_STYLE": "seaborn-v0_8-darkgrid",  # Matplotlib style
    "COLOR_PALETTE": "husl",              # Seaborn color palette
    "FONT_SIZE_TITLE": 14,                # Title font size
    "FONT_SIZE_LABEL": 12,                # Axis label font size
    "FONT_SIZE_TICK": 10,                 # Tick label font size
    
    # === ANALYSIS OPTIONS ===
    "INCLUDE_BAYESIAN_DATA": True,        # Include Bayesian calibration CSV analysis
    "GENERATE_RADAR_CHART": True,         # Generate spider plot
    "GENERATE_HEATMAP": True,             # Generate performance heatmap
    "GENERATE_CORRELATION_MATRIX": True,  # Generate correlation analysis
    "TOP_N_MODELS": 3,                    # Number of top models to highlight
    "VERBOSE": True,                      # Print detailed progress messages
    
    # === STATISTICAL SETTINGS ===
    "CONFIDENCE_LEVEL": 0.95,             # Confidence level for uncertainty bands
    "OUTLIER_THRESHOLD": 3.0,             # Z-score threshold for outlier detection
    "MIN_RUNS_FOR_CORRELATION": 3,        # Minimum runs required for correlation matrix
    
    # === PLANCK TARGET REFERENCES ===
    "PLANCK_TARGET_E": float(os.environ.get("PLANCK_TARGET_E", 0.7619)),
    "PLANCK_TARGET_I": float(os.environ.get("PLANCK_TARGET_I", 0.1309)),
}

