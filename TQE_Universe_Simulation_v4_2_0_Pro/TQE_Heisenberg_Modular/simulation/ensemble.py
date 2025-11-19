# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Stefan Len
#
# ==========================================================================================
# ensemble.py - Initial State Sampling
# ==========================================================================================

import numpy as np

def sample_coherent_states(n, rng, config):
    """Sample coherent state amplitudes with heavy-tailed distribution."""
    mags = np.sqrt(rng.lognormal(
        mean=config["COHERENT_LOG_MEAN"],
        sigma=config["COHERENT_LOG_SIGMA"],
        size=n
    ))
    phases = rng.uniform(0, 2*np.pi, size=n)
    return mags * np.exp(1j * phases)

