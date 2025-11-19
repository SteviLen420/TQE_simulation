# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Stefan Len
#
# Memory management utilities
#
import gc
import matplotlib.pyplot as plt

def cleanup_memory():
    """Clean up memory between phases."""
    gc.collect()
    plt.close('all')

