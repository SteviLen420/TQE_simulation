# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Stefan Len
#
# Formatting utility functions
#
def _fmt(x):
    """Formats a number for clean printing, handling None and non-finite values."""
    return f"{float(x):.4f}" if (x is not None and np.isfinite(x)) else "N/A"

def _pretty_label(s: str) -> str:
    """Converts technical feature names into human-readable labels."""
    base = str(s).strip()
    m = re.match(r"^([A-Za-z_]+)", base)
    if m:
        base = m.group(1)
    base = (base
            .replace("abs_E_minus_I", "|E − I|")
            .replace("logX", "log X")
            .replace("dist_to_goldilocks", "Goldilocks X"))
    return base


