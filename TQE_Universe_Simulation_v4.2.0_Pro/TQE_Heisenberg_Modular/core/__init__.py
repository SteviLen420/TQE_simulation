# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Stefan Len
#
# TQE Heisenberg Modular - Core Package

from .tqe_functions import (
    sample_info_beta, f_lockin, lockin_rate_scale, lockin_potential_scale,
    compute_heisenberg_uncertainty, normalize_coherence
)
from .quantum_system import build_quantum_system
from .information_origin import (
    compute_I_emergent, compute_I_inherent, compute_I_threshold
)

__all__ = [
    'sample_info_beta', 'f_lockin', 'lockin_rate_scale', 'lockin_potential_scale',
    'compute_heisenberg_uncertainty', 'normalize_coherence',
    'build_quantum_system',
    'compute_I_emergent', 'compute_I_inherent', 'compute_I_threshold',
]

