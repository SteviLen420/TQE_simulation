# ===================================================================================
# 03_TQE_(E,I)_UNIVERSE_SIMULATION_imports.py
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
from config import ACTIVE
from io_paths import resolve_output_paths, ensure_colab_drive_mounted
