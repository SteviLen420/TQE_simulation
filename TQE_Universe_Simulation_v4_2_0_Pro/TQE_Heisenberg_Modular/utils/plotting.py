# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Stefan Len
#
# ==========================================================================================
# plotting.py - Plotting Style Setup
# ==========================================================================================

import matplotlib.pyplot as plt

def setup_scientific_plotting_style(config=None):
    """
    Setup clean, scientific plotting style.
    
    Parameters
    ----------
    config : dict or None
        MASTER_CTRL config dict (optional, for font sizes and DPI)
    """
    plt.style.use('default')
    
    # Default values
    fontsize_title = 14
    fontsize_label = 12
    fontsize_legend = 10
    dpi = 300
    
    if config is not None:
        fontsize_title = config.get('PLOT_FONTSIZE_TITLE', fontsize_title)
        fontsize_label = config.get('PLOT_FONTSIZE_LABEL', fontsize_label)
        fontsize_legend = config.get('PLOT_FONTSIZE_LEGEND', fontsize_legend)
        dpi = config.get('PLOT_DPI', dpi)
    
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': 'black',
        'axes.linewidth': 0.8,
        'axes.grid': True,
        'grid.color': 'lightgray',
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
        'font.weight': 'light',
        'font.size': 11,
        'axes.titlesize': fontsize_title,
        'axes.labelsize': fontsize_label,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': fontsize_legend,
        'figure.dpi': dpi,
        'savefig.dpi': dpi,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.2,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

