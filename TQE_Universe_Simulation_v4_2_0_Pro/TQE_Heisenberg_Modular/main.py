# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Stefan Len
#
# ==========================================================================================
# main.py - Main Entry Point
# ==========================================================================================
# TQE Heisenberg Modular: Main entry point for the TQE Heisenberg Fluctuation Simulation
# ==========================================================================================

import sys
import os
import warnings

# Set matplotlib backend BEFORE importing pyplot
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt

# Configure matplotlib for proper PNG generation
plt.ioff()  # Turn off interactive mode
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['savefig.format'] = 'png'
plt.rcParams['figure.max_open_warning'] = 0

warnings.filterwarnings("ignore")

# Check and install packages
from .utils.setup import check_and_install_packages
from .utils.plotting import setup_scientific_plotting_style
from .config import MASTER_CTRL
from .pipeline import run_pipeline

# QuTiP availability check
try:
    from qutip import (basis, destroy, num, coherent, thermal_dm, Qobj, mesolve,
                       mcsolve, expect, enr_thermal_dm, tensor, qeye, entropy_vn)
    from qutip.solver import Options
    QUTIP_AVAILABLE = True
except ImportError:
    print("[WARNING] qutip not available - simulation cannot run")
    QUTIP_AVAILABLE = False

def main(config_override=None):
    """
    Main entry point for TQE Heisenberg Fluctuation Simulation.
    
    Parameters
    ----------
    config_override : dict, optional
        Dictionary to override MASTER_CTRL values
    
    Returns
    -------
    dict
        Results dictionary with all simulation data and metadata
    """
    print("="*80)
    print("TQE HEISENBERG FLUCTUATION SIMULATION v4.2.0 PRO - MODULAR")
    print("="*80)
    print("Theory of the Question of Existence (TQE)")
    print("Local execution only - Desktop output")
    print("="*80)
    
    # Check and install packages
    check_and_install_packages()
    
    # Check QuTiP availability
    if not QUTIP_AVAILABLE:
        raise ImportError("QuTiP is required but not available. Please install it: pip install qutip")
    
    # Setup plotting style
    config = MASTER_CTRL.copy()
    if config_override:
        config.update(config_override)
    setup_scientific_plotting_style(config)
    
    # Run pipeline
    results = run_pipeline(config_override)
    
    return results

if __name__ == "__main__":
    results = main()
    print("\n✅ Pipeline completed successfully!")
    print(f"📁 Results saved to: {results['run_dir']}")

