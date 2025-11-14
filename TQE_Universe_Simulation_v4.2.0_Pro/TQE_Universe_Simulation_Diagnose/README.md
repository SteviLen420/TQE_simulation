SPDX-License-Identifier: MIT

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![Status](https://img.shields.io/badge/status-diagnostics-blueviolet)](#)

# TQE UNIVERSE SIMULATION DIAGNOSE v4.2.0 PRO

**Title:** Integrity & Regression Diagnostics for the TQE Universe Simulation Pipelines  
**Author:** Stefan Len  
**Version:** v4.2.0 PRO

---

## Abstract

`TQE_Universe_Simulation_Diagnose_v4.2.0_Pro.py` is the official health-check suite for the Theory of the Question of Existence (TQE) pipelines. It validates both the monolithic `TQE_Universe_Simulation_Full_Pipeline_v4.2.0_Pro.py` and the modular `TQE_Pipeline_Modular/` implementation, ensuring that 28-phase execution, helper modules, and configuration schemas remain synchronized. The tool performs dependency checks, schema validation, phase signature inspection, optional deep introspection, and an optional smoke run with a handful of universes—then writes structured reports (JSON + CSV) that mirror the wider repository’s output conventions.

---

## Key Capabilities

- **Bidirectional coverage:** Validates both the monolithic and modular pipelines (toggle via CLI flags or `DIAGNOSE_CTRL`).
- **28-phase awareness:** Confirms that every phase function is present and exposes the expected signature, mirroring the production pipeline.
- **Deep component auditing:** Optional `--deep` mode introspects helper modules, PhysicsEngine/PipelineContext APIs, and optional dependencies (healpy, camb, qutip, dynesty, corner).
- **Configuration schema guardrails:** Verifies `MASTER_CTRL` keys, numerical ranges, run modes, and I-definition names before a simulation run is attempted.
- **Smoke test harness:** `--smoke` runs a tiny Monte Carlo sample (default 3 universes × 10 epochs) to exercise key imports, context creation, and helper utilities without the full cost of Phase 1–28.
- **Structured reporting:** Generates timestamped folders under `TQE_Universe_Simulation_Diagnostics/` containing summary JSON/CSV files plus detailed issue/check listings with actionable recommendations.
- **CLI overrides + config dict:** `DIAGNOSE_CTRL` centralizes all toggles, while command-line switches (`--monolithic`, `--modular`, `--all`, `--deep`, `--smoke`) provide quick overrides.

---

## Requirements

- Python **3.9+**
- Core libraries: `numpy`, `pandas`, `matplotlib`, `scipy`, `scikit-learn`, `tqdm`
- Optional (used only when `--deep` or `CHECK_OPTIONAL_DEPS=True`): `healpy`, `camb`, `qutip`, `dynesty`, `corner`
- Runs inside the `TQE_Universe_Simulation_v4.2.0_Pro` tree so it can import both pipeline variants and their helpers.

> Tip: The script only imports heavy optional modules if you enable deep checks; otherwise it stays lightweight and file-system based.

---

## Installation & Workspace Setup

```bash
# 1. Clone the repository
git clone <TQE_simulation repo>
cd TQE_simulation/TQE_Universe_Simulation_v4.2.0_Pro

# 2. (Optional but recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install core dependencies
pip install -U pip wheel setuptools
pip install numpy pandas matplotlib scipy scikit-learn tqdm

# 4. Install optional deps only if you plan to run --deep
pip install healpy camb qutip dynesty corner
```

---

## Usage

From the `TQE_Universe_Simulation_v4.2.0_Pro/TQE_Universe_Simulation_Diagnose/` directory:

```bash
python TQE_Universe_Simulation_Diagnose_v4.2.0_Pro.py              # Default: check both pipelines
python TQE_Universe_Simulation_Diagnose_v4.2.0_Pro.py --monolithic # Only full pipeline
python TQE_Universe_Simulation_Diagnose_v4.2.0_Pro.py --modular    # Only modular pipeline
python TQE_Universe_Simulation_Diagnose_v4.2.0_Pro.py --all        # Explicitly both (default)
python TQE_Universe_Simulation_Diagnose_v4.2.0_Pro.py --deep       # Enable optional/heavy checks
python TQE_Universe_Simulation_Diagnose_v4.2.0_Pro.py --smoke      # Run tiny Monte Carlo smoke test
```

Multiple switches can be combined (e.g., `--modular --deep --smoke`).

---

## Configuration (`DIAGNOSE_CTRL`)

All behavior is driven by the `DIAGNOSE_CTRL` dictionary near the top of the script. CLI flags override these settings at runtime. The most common keys are:

| Key | Default | Description |
| --- | --- | --- |
| `CHECK_MONOLITHIC` / `CHECK_MODULAR` / `CHECK_BOTH` | `True` | Select which pipeline(s) to inspect |
| `DEEP_CHECK` | `False` | Enables optional dependency checks, helper introspection, and full imports |
| `RUN_SMOKE_TEST` | `False` | Executes a miniature Monte Carlo run (configurable universes/epochs) |
| `CHECK_DEPENDENCIES` | `True` | Verifies essential Python packages are importable |
| `CHECK_CONFIG` | `True` | Validates `MASTER_CTRL` schema and numeric ranges |
| `CHECK_PHASES` / `CHECK_FUNCTIONS` | `True` | Ensures all 28 phases and key helper APIs exist |
| `CHECK_OPTIONAL_DEPS` | `False` | Only triggered when `DEEP_CHECK` is `True` |
| `SAVE_JSON` / `SAVE_CSV` | `True` | Persist summary + detailed reports under the diagnostics directory |
| `OUTPUT_DIR_PREFIX` | `"TQE_Universe_Simulation_Diagnostics"` | Root folder auto-created on Desktop or cwd |

Adjust the dictionary for persistent defaults, or rely on CLI switches for ad-hoc runs.

---

## Diagnostic Workflow

The tool organizes its work into conceptual groups (mirroring the broader pipeline’s phase mindset):

| Group | Description |
| --- | --- |
| **Group 1 – Dependencies & Config** | Validates Python requirements and the `MASTER_CTRL` schema (run modes, I-definitions, numeric ranges, optional keys). |
| **Group 2 – Pipeline Structure** | File-level checks, syntax validation, and phase/function signature audits for both pipeline variants. Optional `--deep` adds helper, PhysicsEngine, and PipelineContext introspection. |
| **Group 2.9 – Phase Integration** | Ensures that `run_pipeline` stitches every phase together (monolithic & modular). |
| **Group 3 – Smoke Test** | (Optional) Runs a miniature simulation to guarantee that imports, contexts, and helper utilities execute end-to-end without generating full outputs. |
| **Group 4 – Reporting (Phase 9)** | Saves summary JSON/CSV files plus detailed issue/check logs and prioritized recommendations. |

Each `ok()/warn()/err()` call appends an entry to the `DiagnosticReport`, which is later converted into machine-readable artifacts.

---

## Output Structure

Diagnostics mirror the main pipeline’s timestamped layout. By default results land under your Desktop (fallback: current working directory):

```
~/Desktop/TQE_Universe_Simulation_Diagnostics/
└── TQE_Universe_Simulation_Diagnostics_<timestamp>/
    ├── diagnostic_results.json            # Summary (CLI-friendly)
    ├── diagnostic_results.csv             # Quick status table
    ├── diagnostic_report_full.json        # Structured issues/checks/recommendations
    ├── diagnostic_report_issues.csv       # One row per issue (severity, component, suggestion)
    ├── diagnostic_report_checks.csv       # One row per check (status, phase, pipeline)
    ├── diagnostic_report_summary_by_category.csv
    └── diagnostic_report_recommendations.json
```

Use these files in CI dashboards, regression monitoring, or when filing GitHub issues.

---

## Smoke Test Details

- Default size: `SMOKE_TEST_UNIVERSES=3`, `SMOKE_TEST_EPOCHS=10`
- Creates a temporary `PipelineContext`, evaluates the `PhysicsEngine`, calls Monte Carlo helpers, and exercises plotting/formatting/memory utilities.
- Logs per-step successes/failures and reports aggregated status at the end of the smoke block.
- Ideal for pre-commit sanity checks after refactoring a phase or helper module.

> Warning: the smoke test still constructs CMB helpers and entropy panels, so it expects the same dependencies as a normal run (just smaller data).

---

## Interpreting the Reports

- **`diagnostic_results.json/csv`** – quick PASS/FAIL overview with flags indicating which checks ran.
- **`diagnostic_report_full.json`** – hierarchical structure used by downstream tooling; contains issues grouped by severity/category plus recommendations.
- **Issues CSV** – filterable table (id, severity, category, component, suggestion, phase, pipeline type).
- **Recommendations JSON** – prioritized hints (HIGH/MEDIUM/LOW) based on the captured issues (e.g., missing dependency, malformed MASTER_CTRL value, absent phase function).

---

## Extending the Diagnostics

- Add new checks by appending helper functions and logging via `ok()/warn()/err()` so they automatically populate the report.
- Update `DIAGNOSE_CTRL` defaults when new optional packages or configuration keys are introduced.
- Use `reset_diagnostic_report()` when embedding the diagnostics in larger test harnesses to avoid cross-run contamination.

---

## Contact & Support

- **Author:** Stefan Len  
- **Email:** stefan@tqe-theory.space  
- **Repository:** TODO – add GitHub link when available

For bug reports, include the generated diagnostic folder (JSON/CSV files) so issues can be reproduced quickly.

---

## Citation

```bibtex
@software{Len_2025_TQE_Diagnose_v4_2,
  author    = {Len, Stefan},
  title     = {{TQE Universe Simulation Diagnose v4.2.0 PRO}},
  version   = {4.2.0},
  year      = {2025},
  publisher = {GitHub},
  url       = {https://github.com/<TQE_repo>},
}
```

---

All contributions, bug reports, or feature suggestions that improve the reliability of the TQE pipelines are welcome. Run the diagnostics any time a dependency is upgraded, new phases are added, or the modular architecture changes—then share the report alongside your findings. Happy debugging! 🛠️

