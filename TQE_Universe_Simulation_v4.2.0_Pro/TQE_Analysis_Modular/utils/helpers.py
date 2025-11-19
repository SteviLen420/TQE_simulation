# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Stefan Len
#
# Helper utility functions

def extract_i_definition(dirname: str) -> str:
    """Extract I-definition name from directory name."""
    if dirname.startswith("Eonly_"):
        return "energy_only"
    elif dirname.startswith("EplusI_"):
        # EplusI_kl_divergence_20251030_223511 → kl_divergence
        parts = dirname.split("_")
        # Find the timestamp part (8 digits)
        for i, part in enumerate(parts):
            if part.isdigit() and len(part) == 8:
                # Everything before timestamp is the I-definition
                return "_".join(parts[1:i])
        return "unknown"
    return "unknown"

