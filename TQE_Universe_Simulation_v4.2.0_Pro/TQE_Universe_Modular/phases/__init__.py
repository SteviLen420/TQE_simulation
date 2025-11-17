# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Stefan Len

# Import all phase functions
from .phase_01_10 import (
    phase_02_stability_curve,
    phase_03_scatter_ei,
    phase_04_fluctuation_panels,
    phase_05_stability_by_i,
    phase_06_lockin_histogram,
    phase_07_stability_distribution,
    phase_08_avg_lockin_curve,
    phase_09_feature_importance,
    phase_10_emergent_laws,
)

from .phase_11_20 import (
    phase_11_finetuning_detector,
    phase_12_best_universe_plots,
    phase_13_generate_missing_cmb_maps,
    phase_14_entropy_volatility,
    phase_15_planck_validation,
    phase_16_cmb_anomaly_detection,
    phase_17_ei_importance_comparison,
    phase_18_multi_mode_goldilocks_comparison,
    phase_19_cmb_analysis_plots,
    phase_20_comprehensive_correlation_analysis,
)

from .phase_21_28 import (
    phase_21_advanced_statistical_analysis,
    phase_22_cmb_anomaly_analysis_plots,
    phase_23_enhanced_physics_analysis,
    phase_24_comprehensive_data_extraction,
    phase_25_advanced_anomaly_detection,
    phase_26_advanced_law_detection,
    phase_27_comprehensive_visualization_extraction,
    phase_28_final_summary,
    integrate_complexity_analysis,
)

__all__ = [
    'phase_02_stability_curve',
    'phase_03_scatter_ei',
    'phase_04_fluctuation_panels',
    'phase_05_stability_by_i',
    'phase_06_lockin_histogram',
    'phase_07_stability_distribution',
    'phase_08_avg_lockin_curve',
    'phase_09_feature_importance',
    'phase_10_emergent_laws',
    'phase_11_finetuning_detector',
    'phase_12_best_universe_plots',
    'phase_13_generate_missing_cmb_maps',
    'phase_14_entropy_volatility',
    'phase_15_planck_validation',
    'phase_16_cmb_anomaly_detection',
    'phase_17_ei_importance_comparison',
    'phase_18_multi_mode_goldilocks_comparison',
    'phase_19_cmb_analysis_plots',
    'phase_20_comprehensive_correlation_analysis',
    'phase_21_advanced_statistical_analysis',
    'phase_22_cmb_anomaly_analysis_plots',
    'phase_23_enhanced_physics_analysis',
    'phase_24_comprehensive_data_extraction',
    'phase_25_advanced_anomaly_detection',
    'phase_26_advanced_law_detection',
    'phase_27_comprehensive_visualization_extraction',
    'phase_28_final_summary',
    'integrate_complexity_analysis',
]

