# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Stefan Len
#
# Plotting utility functions
#
import matplotlib.pyplot as plt
from ..config.master_ctrl import MASTER_CTRL

def setup_scientific_plotting_style(config=None):
    """Setup clean, scientific plotting style with consistent fonts and readability using MASTER_CTRL parameters."""
    if config is None:
        config = MASTER_CTRL
    
    plt.style.use('default')

    # Set global matplotlib parameters for scientific appearance using MASTER_CTRL
    plt.rcParams.update({
        # Figure and axes
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': 'black',
        'axes.linewidth': config.get('PLOT_EDGE_LINEWIDTH', 0.8),
        'axes.grid': True,
        'grid.color': 'lightgray',
        'grid.alpha': config.get('PLOT_GRID_ALPHA', 0.3),
        'grid.linewidth': config.get('PLOT_GRID_LINEWIDTH', 0.5),

        # Fonts and text - UNIFIED STYLE (THINNER FONTS)
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif'],
        'font.weight': 'light',  # Thinner font weight
        'font.size': 11,  # Slightly smaller base size
        'axes.titlesize': config.get('PLOT_FONTSIZE_TITLE', 14),  # Smaller title
        'axes.labelsize': config.get('PLOT_FONTSIZE_LABEL', 12),  # Smaller labels
        'xtick.labelsize': 10,  # Smaller tick labels
        'ytick.labelsize': 10,  # Smaller tick labels
        'legend.fontsize': config.get('PLOT_FONTSIZE_LEGEND', 10),  # Smaller legend
        'figure.titlesize': 16,  # Smaller figure title

        # Colors
        'axes.prop_cycle': plt.cycler('color', config.get('PLOT_COLOR_CYCLE', ['#87CEEB', '#FA8072', '#98FB98', '#DDA0DD', '#F0E68C', '#FFB6C1', '#20B2AA'])),

        # Layout - PUBLICATION QUALITY
        'figure.dpi': config.get('PLOT_DPI', 300),  # High DPI for sharp display
        'savefig.dpi': config.get('PLOT_SAVE_DPI', 300),  # 300 DPI for publication
        'savefig.bbox': 'tight',  # Prevent label cutoff
        'savefig.pad_inches': 0.2,  # More padding to prevent overlap (was: 0.15)

        # Spines
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.left': True,
        'axes.spines.bottom': True,

        # Ticks
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.size': 5,
        'ytick.major.size': 5,
        'xtick.minor.size': 3,
        'ytick.minor.size': 3,
    })

def apply_consistent_plot_style(ax, title="", xlabel="", ylabel="", config=None):
    """Apply consistent styling to any plot - SYNCHRONIZED with Goldilocks PNG style."""
    if config is None:
        config = MASTER_CTRL
    
    # SYNCHRONIZED FONTSIZES (matching Goldilocks PNG style)
    title_size = 18  # Consistent with all other PNG titles
    label_size = 16  # Consistent with all other PNG labels
    tick_size = 13   # Consistent with all other PNG ticks
    
    grid_alpha = config.get('PLOT_GRID_ALPHA', 0.3)
    grid_linewidth = config.get('PLOT_GRID_LINEWIDTH', 0.5)
    edge_linewidth = config.get('PLOT_EDGE_LINEWIDTH', 0.8)
    
    if title:
        ax.set_title(title, fontsize=title_size, pad=20)  # Normal weight (no bold/light)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=label_size)  # Normal weight
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=label_size)  # Normal weight
    
    # Set tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=tick_size)
    ax.tick_params(axis='both', which='minor', labelsize=tick_size-1)
    
    # Ensure grid is visible
    ax.grid(True, alpha=grid_alpha, linestyle='-', linewidth=grid_linewidth)
    ax.set_axisbelow(True)
    
    # Make spines more visible
    for spine in ax.spines.values():
        spine.set_linewidth(edge_linewidth)

