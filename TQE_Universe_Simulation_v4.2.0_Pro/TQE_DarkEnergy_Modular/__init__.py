# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Stefan Len
#
# TQE_DarkEnergy_Modular - Modularized TQE Dark Energy Coupling Simulation
# Based on the Theory of the Question of Existence (TQE)
#
# This package provides a modular implementation of the TQE Dark Energy Coupling Simulation,
# organized into logical components for better maintainability and extensibility.

__version__ = "4.2.0"
__author__ = "Stefan Len"

# Import main components for easy access
from .config import COSMO_PARAMS, MASTER_CTRL, FIDUCIAL_PARAMS, c_light, G_newton, M_planck
from .utils import (
    install_package,
    check_and_install_all_packages,
    cleanup_memory,
    apply_performance_mode,
    setup_google_drive_automatically,
    check_google_drive_status,
    set_deterministic_seed,
    save_reproducibility_snapshot,
    w_eff_CPL,
    _top_hat_W
)
from .tqe_core import EnergyInformationContent, CouplingModel
from .cosmology import FriedmannEvolution
from .data_loader import (
    load_pantheon_plus_data,
    load_boss_bao_data,
    load_planck_cmb_data,
    PlanckCMBDataLoader,
    CMBPlanckValidation
)
from .observables import ObservablePredictions
from .inference import BayesianInferenceEngine
from .structure import GalaxyStructureAnalyzer
from .simulation import (
    TQEDarkEnergyCouplingSimulation,
    run_sanity_checks,
    get_file_prefix,
    add_prefix_to_path,
    GalaxyStructureAnalysis
)
from .visualization import (
    compare_eonly_vs_eplusi,
    compute_bayes_factors_all_models,
    create_bayes_factor_plot,
    create_eonly_vs_eplusi_dashboard,
    find_goldilocks_zone_bayesian
)
from .pipeline import run_automatic_tqe_darkenergy_pipeline, run_integrated_aggregator
from .main import main

__all__ = [
    # Config
    'COSMO_PARAMS',
    'MASTER_CTRL',
    'FIDUCIAL_PARAMS',
    'c_light',
    'G_newton',
    'M_planck',
    # Utils
    'install_package',
    'check_and_install_all_packages',
    'cleanup_memory',
    'apply_performance_mode',
    'setup_google_drive_automatically',
    'check_google_drive_status',
    'set_deterministic_seed',
    'save_reproducibility_snapshot',
    'w_eff_CPL',
    '_top_hat_W',
    # Core
    'EnergyInformationContent',
    'CouplingModel',
    # Cosmology
    'FriedmannEvolution',
    # Data
    'load_pantheon_plus_data',
    'load_boss_bao_data',
    'load_planck_cmb_data',
    'PlanckCMBDataLoader',
    'CMBPlanckValidation',
    # Observables
    'ObservablePredictions',
    # Inference
    'BayesianInferenceEngine',
    # Structure
    'GalaxyStructureAnalyzer',
    # Simulation
    'TQEDarkEnergyCouplingSimulation',
    'run_sanity_checks',
    'get_file_prefix',
    'add_prefix_to_path',
    'GalaxyStructureAnalysis',
    # Visualization
    'compare_eonly_vs_eplusi',
    'compute_bayes_factors_all_models',
    'create_bayes_factor_plot',
    'create_eonly_vs_eplusi_dashboard',
    'find_goldilocks_zone_bayesian',
    # Pipeline
    'run_automatic_tqe_darkenergy_pipeline',
    'run_integrated_aggregator',
    # Main
    'main',
]

