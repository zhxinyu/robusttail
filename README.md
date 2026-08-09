# Robusttail

## Table of contents

- [Quick start](#quick-start)
- [Reproducing the paper](#reproducing-the-paper)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Platform support](#platform-support)
  - [Recommended installation: Conda environment files](#recommended-installation-conda-environment-files)
  - [R Installation](#r-installation)
  - [Mosek Academic Licenses](#mosek-academic-licenses)
  - [Verification](#verification)
  - [Troubleshooting](#troubleshooting)

---

## Quick start

After installation, from the repo root run a short tail-probability experiment:

```bash
cd experiments/run_scripts
PYTHONPATH="$(git rev-parse --show-toplevel)" python exp_tail_probability.py --experiment quick_run
```

(If not in a git repo, set `PYTHONPATH` to the project root instead.)

## Reproducing the paper

To regenerate the tables and figures reported in the paper, follow the
[paper reproduction guide](experiments/reproduction/README.md). It lists
the command for each experimental display and the corresponding generated
LaTeX table or plot.

## Installation

### Prerequisites

- [Conda or Miniconda](https://docs.conda.io/projects/miniconda/en/latest/)

The supplied environment files provide the supported Python 3.12 and R 4.2
setup for this project.

### Platform support

- **Supported / tested**: **macOS (ARM64)** and **Linux via WSL (Ubuntu/WSL2)**.
- **Windows**: **not supported natively** at the moment. Please use **WSL**.

### Recommended installation: Conda environment files

The platform-specific YAML files are the maintained dependency specifications.

1. **Create the conda environment from the platform-specific file:**
   - **Linux / WSL:** `conda env create -f environment_linux.yml`
   - **macOS (ARM64):** `conda env create -f environment_osx-arm64.yml`

2. **Activate the environment:**

   ```bash
   conda activate rs
   ```

3. **Verify installation:**

   ```bash
   python --version   # Should show Python 3.12
   R --version       # Should show R version
   ```

### R Installation

R and its packages are managed within the active Conda environment; no separate
system-wide R installation is needed. If a benchmark package is unavailable
through Conda on a platform, the benchmark bridge installs it from CRAN when
first imported.

### Mosek Academic Licenses

For academic use, obtain a free license at [Mosek Academic Licenses](https://www.mosek.com/products/academic-licenses/).

After obtaining the license:

1. Download the license file.
2. Place it in:
   - **Linux / macOS:** `~/mosek/mosek.lic`
3. Verify installation:

   ```python
   import mosek
   print(mosek.Env.getversion())
   ```

### Verification

Test that everything is installed correctly:

```bash
python -c "import matplotlib, mosek, numpy, pandas, PIL, rpy2, scipy, tqdm; print('Python packages available')"
python -c "from experiments.run_scripts.tail_probability import benchmark_tail_probability_estimation; print('R benchmark packages available')"
```

### Troubleshooting

- **R/rpy2 issues:** Ensure R is properly installed and `R_HOME` is set correctly.
- **Mosek license:** Check that the license file is in the correct location and not expired.
- **Platform-specific issues:** Use the appropriate environment file for your platform.
