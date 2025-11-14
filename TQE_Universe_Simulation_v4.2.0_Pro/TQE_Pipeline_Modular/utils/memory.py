# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Stefan Len
#
# Memory management utilities
#
import gc
import matplotlib.pyplot as plt

def optimize_for_colab():
    """Apply Colab-specific optimizations."""
    import gc
    
    # Clear any existing plots to free memory
    plt.close('all')
    
    # Force garbage collection
    gc.collect()
    
    # Set matplotlib to use less memory
    plt.rcParams['figure.max_open_warning'] = 0
    
    print("[COLAB] Applied memory optimizations")

def cleanup_memory():
    """Clean up memory between phases."""
    gc.collect()
    plt.close('all')

