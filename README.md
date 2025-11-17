SPDX-License-Identifier: MIT

Copyright (c) 2025 Stefan Len

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17627756.svg)](https://doi.org/10.5281/zenodo.17627756)
[![arXiv](https://img.shields.io/badge/arXiv-coming%20soon-b31b1b.svg)](#)
[![GitHub stars](https://img.shields.io/github/stars/SteviLen420/TQE_simulation?style=social)](https://github.com/SteviLen420/TQE_simulation)
[![GitHub forks](https://img.shields.io/github/forks/SteviLen420/TQE_simulation?style=social)](https://github.com/SteviLen420/TQE_simulation)
[![Research Status](https://img.shields.io/badge/status-active%20research-green)](https://github.com/SteviLen420/TQE_simulation)

# Theory of The Question of Existence (TQE)

**Investigating the Emergence of Physical Laws from Energy-Information Coupling**

**Author**: Stefan Len

## Overview

The Theory of The Question of Existence (TQE) is a theoretical framework proposing that stable physical laws emerge from the primitive coupling of Energy (E) and Information (I). Rather than being axiomatic, physical constants and laws are hypothesized to dynamically stabilize through a probabilistic selection mechanism during cosmogenesis.

This repository contains computational research exploring this hypothesis through Monte Carlo simulations, mathematical modeling, and analysis of cosmological observables.

## Core Hypothesis

TQE posits that:

- Physical laws are not pre-existing but emerge from quantum fluctuations in a pre-law state
- Stability arises from the coupling of vacuum energy fluctuations with an information-theoretic orientation parameter
- This coupling creates a "Goldilocks window" where law-governed universes can stabilize
- The framework yields falsifiable predictions about statistical anomalies in cosmological observations

At the mathematical level the hypothesis is encoded in a modulation rule for the primordial probability distribution $P(\psi)$ of universal states:

$$
P'(\psi)=P(\psi)\cdot f(E,I),
$$

where $E$ denotes a vacuum fluctuation energy sample (drawn from a heavy-tailed distribution), $I$ is an information-oriented asymmetry measure, and $f(E,I)$ biases collapse toward law-consistent outcomes. The baseline implementation uses

$$
f(E,I)=\exp\!\left(-\frac{(E-E_c)^2}{2\sigma^2}\right)\left(1+\alpha I\right),
$$

with $E_c$ the Goldilocks center, $\sigma$ the stability width, and $\alpha$ the strength of the informational bias. The model tracks a composite complexity parameter $X=E\cdot I$ (stability gate) together with the asymmetry $|E-I|$ (lock-in trigger) to decide whether a universe reaches the late-time law-lock-in state.

The information parameter itself is defined operationally. In simulations it is computed from consecutive probability distributions $P_t$ and $P_{t+1}$ via a normalized Kullback–Leibler divergence (optionally fused with Shannon entropy):

$$
I = \frac{D_{\mathrm{KL}}(P_t \parallel P_{t+1})}{1 + D_{\mathrm{KL}}(P_t \parallel P_{t+1})},
$$

ensuring $0 \le I \le 1$. A universe is deemed to have stabilized its laws when the relative change in key observables satisfies $\Delta P/P < 5\times 10^{-3}$ over at least six consecutive epochs. Together these prescriptions make the TQE hypothesis quantitative, reproducible, and falsifiable within Monte Carlo ensembles.

## Scientific Context

TQE addresses fundamental questions in cosmology:

- Why do stable, complexity-permitting physical laws exist?
- How might physical constants be dynamically selected rather than fixed?
- What mechanisms could explain fine-tuning without invoking anthropic reasoning alone?

The framework provides a quantitative, testable approach to these questions through computational simulation and mathematical analysis.

## Research Approach

The work explores the TQE hypothesis through:

- Monte Carlo simulations of universe ensembles
- Analysis of law stabilization dynamics
- Generation and study of cosmological observables
- Statistical investigation of large-scale anomalies
- Integration of explainable AI methods for pattern discovery

## Citation

If you use this work in your research, please cite:

Author ORCID (per `CITATION.cff`): https://orcid.org/0009-0007-0383-7315

```bibtex
@software{Len_2025_TQE,
  author    = {Len, Stefan},
  orcid     = {https://orcid.org/0009-0007-0383-7315},
  title     = {{Theory of The Question of Existence (TQE)}},
  year      = {2025},
  publisher = {Zenodo},
  url       = {https://github.com/SteviLen420/TQE_simulation},
  doi       = {10.5281/zenodo.17627756}
}
```

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

## Contact

For questions, collaborations, or feedback:

**Email**: stefan@tqe-theory.space

---

*This is an active research project. Contributions, feedback, and scientific collaboration are welcome.*
