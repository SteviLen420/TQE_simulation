# ===================================================================================
# TQE_03_EI_UNIVERSE_SIMULATION_imports.py
# ===================================================================================
# Author: Stefan Len
# ===================================================================================

# ---------------------------
# Standard Python libraries
# ---------------------------
import os
import sys
import math
import time
import json
import random
import pathlib
from copy import deepcopy

# ---------------------------
# Numerical & data handling
# ---------------------------
import numpy as np
import pandas as pd

# ---------------------------
# Visualization
# ---------------------------
import matplotlib.pyplot as plt

# ---------------------------
# Scientific & statistical tools
# ---------------------------
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon
from scipy.interpolate import UnivariateSpline

# ---------------------------
# Quantum-specific (optional, if needed)
# ---------------------------
try:
    import qutip as qt
except ImportError:
    qt = None
    print("[WARN] QuTiP not installed. Quantum features disabled.")

# ---------------------------
# Machine Learning & XAI
# ---------------------------
try:
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    import shap
    import lime
    import lime.lime_tabular
except ImportError:
    print("[WARN] Some ML/XAI libraries are missing. XAI features may be disabled.")

# ---------------------------
# Internal project modules
# ---------------------------
from TQE_00_EI_UNIVERSE_SIMULATION_config import ACTIVE
from TQE_02_EI_UNIVERSE_SIMULATION_Output_io_paths import resolve_output_paths, ensure_colab_drive_mounted

# ---------------------------
# I/O path bootstrap (one-time)
# ---------------------------
try:
    # Colab Drive mount (no-op desktopon)
    ensure_colab_drive_mounted(ACTIVE)
except Exception as e:
    print("[WARN] Drive mount skipped:", e)

# Resolve and cache run paths once per process (stable run_id)
PATHS = resolve_output_paths(ACTIVE)

# Convenience aliases used by stages
RUN_DIR = PATHS["primary_run_dir"]
FIG_DIR = PATHS["fig_dir"]
