# Robusttail

## Table of contents

- [Experiments](#experiments)
  - [Setup](#setup)
  - [Scripts](#scripts)
  - [Summary](#summary)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Environment Setup](#environment-setup)
    - [Option 1: Using Conda Environment (Recommended)](#option-1-using-conda-environment-recommended)
    - [Option 2: Manual Installation](#option-2-manual-installation)
  - [R Installation](#r-installation)
  - [Mosek Academic Licenses](#mosek-academic-licenses)
  - [Verification](#verification)
  - [Troubleshooting](#troubleshooting)

---

## Experiments

Run scripts for tail probability estimation, quantile estimation, and benchmarks.

All commands below assume you are in the **`run_scripts`** directory with the **project root on `PYTHONPATH`**.

### Setup

1. **Add the project root to `PYTHONPATH`** (so imports like `droevt`, `tail_probability` resolve).

   ```bash
   export PYTHONPATH=/path/to/robusttail
   ```

2. **Use `run_scripts` as the working directory.**

   ```bash
   cd experiments/run_scripts
   ```

**Example run** (tail probability):

```bash
PYTHONPATH=/path/to/robusttail python exp_tail_probability.py --experiment thresholds
```

---

### Scripts

#### 1. `exp_tail_probability.py` — DRO tail probability experiments

Runs **distributionally robust (DRO) tail probability estimation** on synthetic and real (CMT) data: multiple distributions, threshold percentages, objective functions, and scarce data.

| Option | Description |
| :----- | :---------- |
| `--experiment thresholds` | Vary threshold percentages (single and multi-threshold). |
| `--experiment percentage_lhs` | Vary left quantile level (0.9–0.99) in the objective function. |
| `--experiment scarce_data` | Scarce synthetic data (n=30). |
| `--experiment real_data` | Real CMT earthquake data, multiple regions and thresholds. |

**How to run:**

```bash
PYTHONPATH=/path/to/robusttail python exp_tail_probability.py --experiment thresholds
```

---

#### 2. `exp_quantitle_estimation.py` — DRO quantile estimation

Runs **DRO quantile estimation** (ellipsoidal ambiguity) on synthetic data: gamma, lognorm, pareto; multiple threshold percentages (single and multi).

**How to run:**

```bash
PYTHONPATH=/path/to/robusttail python exp_quantitle_estimation.py
```

---

#### 3. `exp_cmt.py` — Real data (CMT) vs EVT

Compares **DRO tail probability** with **EVT-based methods** on parsed CMT (earthquake) data. Lets you vary threshold percentage, confidence level, or run bootstrap estimation.

| Option | Description |
| :----- | :---------- |
| `--function run_different_threshold_percentages` | Vary threshold percentage (default). |
| `--function run_different_critical_values` | Vary critical values. |
| `--function run_different_confidence_levels` | Vary confidence levels. |
| `--function run_bootstrap_estimation` | Bootstrap estimation. |

**How to run:**

```bash
PYTHONPATH=/path/to/robusttail python exp_cmt.py --function run_different_threshold_percentages
```

---

#### 4. `exp_bound_support.py` — Bounded-support (Gen Pareto) experiment

Runs **DRO tail probability with bounded right endpoint** on **generalized Pareto** data: multiple quantiles (0.95, 0.99, 0.995) and right endpoints.

**How to run:**

```bash
PYTHONPATH=/path/to/robusttail python exp_bound_support.py
```

---

#### 5. `exp_benchmark_run.py` — EVT benchmark (synthetic + real)

Runs **traditional EVT tail probability methods** (POT, POT bootstrap, PL, Bayesian, PWM) on synthetic or real CMT data for comparison with DRO. **`--exp` is required.**

| Option | Description |
| :----- | :---------- |
| `--exp synthetic` | Synthetic data only (gamma, lognorm, pareto, genpareto; n=500; percentiles 0.9, 0.95, 0.99). |
| `--exp real` | Real CMT data only (multiple regions and left-endpoint thresholds). |

**How to run:**

```bash
PYTHONPATH=/path/to/robusttail python exp_benchmark_run.py --exp synthetic
PYTHONPATH=/path/to/robusttail python exp_benchmark_run.py --exp real
```

---

### Summary

| Script | Purpose |
| :----- | :------ |
| `exp_tail_probability.py` | DRO tail probability (synthetic + real); `--experiment` chooses which experiment. |
| `exp_quantitle_estimation.py` | DRO quantile estimation on synthetic data; no args. |
| `exp_cmt.py` | CMT real data: DRO vs EVT; `--function` chooses which experiment. |
| `exp_bound_support.py` | Bounded-support (Gen Pareto) tail probability; no args. |
| `exp_benchmark_run.py` | EVT benchmark (synthetic or real); `--exp` required: `synthetic` or `real`. |

---

## Installation

### Prerequisites

- [Conda](https://docs.conda.io/en/latest/miniconda.html) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- Python 3.12
- R (for R-based experiments)

### Environment Setup

#### Option 1: Using Conda Environment (Recommended)

1. **Create the conda environment from the base environment file:**

   ```bash
   conda env create -f environment_base.yml
   ```

2. **Or use the platform-specific file:**
   - **macOS (ARM64):** `conda env create -f environment_osx-arm64.yml`

3. **Activate the environment:**

   ```bash
   conda activate rs
   ```

4. **Verify installation:**

   ```bash
   python --version   # Should show Python 3.12
   R --version       # Should show R version
   ```

#### Option 2: Manual Installation

If you prefer to install packages manually:

```bash
# Create a new conda environment
conda create -n rs python=3.12

# Activate the environment
conda activate rs

# Install core dependencies
conda install -c conda-forge numpy pandas scipy r-base=4.2 rpy2

# Install Python packages via pip
pip install matplotlib mosek
```

### R Installation

R is included in the conda environment. For a system-wide R installation, see [CRAN](https://cran.r-project.org/).

### Mosek Academic Licenses

For academic use, obtain a free license at [Mosek Academic Licenses](https://www.mosek.com/products/academic-licenses/).

After obtaining the license:

1. Download the license file.
2. Place it in:
   - **Linux/macOS:** `~/.mosek/mosek.lic`
   - **Windows:** `%USERPROFILE%\mosek\mosek.lic`
3. Verify installation:

   ```python
   import mosek
   print(mosek.Env.getversion())
   ```

### Verification

Test that everything is installed correctly:

```bash
python -c "import numpy, pandas, scipy, matplotlib, mosek, rpy2; print('All packages imported successfully')"
```

### Troubleshooting

- **R/rpy2 issues:** Ensure R is properly installed and `R_HOME` is set correctly.
- **Mosek license:** Check that the license file is in the correct location and not expired.
- **Platform-specific issues:** Use the appropriate environment file for your platform.
