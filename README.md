# Run Experiments

```bash
## To run all tasks.
bash testRunOnAllTasks.sh

## To run in parallel tail probability estimation problem with single threshold:
python3 tailProbabilityEstimationSingleThreshold.py

## To run in parallel tail probability estimation problem with multiple thresholds:
python3 tailProbabilityEstimationMultipleThresholds.py

## To run single tail probabiltiy estimation:
python3 tailProbabilityEstimationUnit.py

## To run on quantile estimation--single threshold.
python3 quantileEstimationSingleThreshold.py

## To run on quantile estimation--multiple thresholds.
python3 quantileEstimationMultipleThresholds.py

## To run single quantile estimation:
python3 quantileEstimationUnit.py

```

# Installation

## Prerequisites

- [Conda](https://docs.conda.io/en/latest/miniconda.html) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- Python 3.12
- R (for R-based experiments)

## Environment Setup

### Option 1: Using Conda Environment (Recommended)

1. **Create the conda environment from the base environment file:**
   ```bash
   conda env create -f environment_base.yml
   ```

2. **Or use the platform-specific file:**
   - **macOS (ARM64):** `conda env create -f environment_osx-arm64.yml`
   - **Linux:** `conda env create -f environment_linux.yml` (TBD)

3. **Activate the environment:**
   ```bash
   conda activate rs
   ```

4. **Verify installation:**
   ```bash
   python --version  # Should show Python 3.12
   R --version       # Should show R version
   ```

### Option 2: Manual Installation

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

## R Installation

R is included in the conda environment. If you need a system-wide R installation, follow instructions at: https://cran.rstudio.com

## Mosek Academic Licenses

For academic use, obtain a free license at: https://www.mosek.com/products/academic-licenses/

After obtaining the license:
1. Download the license file
2. Place it in `~/.mosek/mosek.lic` (Linux/macOS) or `%USERPROFILE%\mosek\mosek.lic` (Windows)
3. Verify installation:
   ```python
   import mosek
   print(mosek.Env.getversion())
   ```

## Verification

Test that everything is installed correctly:

```bash
python -c "import numpy, pandas, scipy, matplotlib, mosek, rpy2; print('All packages imported successfully')"
```

## Troubleshooting

- **R/rpy2 issues:** Ensure R is properly installed and `R_HOME` is set correctly
- **Mosek license:** Check that the license file is in the correct location and not expired
- **Platform-specific issues:** Use the appropriate environment file for your platform
