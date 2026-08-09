# Reproducing the manuscript experiments

The commands below regenerate the experimental tables and figures in the
manuscript. Run them from the repository root. Generated files are written
under `experiments/generated/` by default. This directory is intentionally
excluded from version control: each command creates its documented numerical
outputs plus a manuscript-ready LaTeX row fragment or plot artifact at run
time. Pass `--output-dir PATH` to a direct command to use another local
folder; the runner creates it when needed.

Except where the stages are shown separately, the `all` stage performs the
complete generation and rendering workflow in one command.

Choose one local output root before running the commands below:

```bash
OUTPUT_ROOT="$PWD/reproduced-results"
mkdir -p "$OUTPUT_ROOT"
```

Each study writes only within its named subdirectory of `OUTPUT_ROOT`.

## Environment

Create and activate the supplied `rs` Conda environment:

```bash
conda env create -f environment_linux.yml
conda activate rs
```

On Apple Silicon, use `environment_osx-arm64.yml` instead. The optimization
experiments require a valid MOSEK license; installation details are in the
repository's main `README.md`.

The simulation runners below use 16 workers. A worker is an independent Python
process that handles one simulation task at a time; this setting controls CPU
parallelism, not the number of Monte Carlo repetitions. A smaller value can be
passed with `--workers` when fewer CPU cores are available. Long-running
simulations write an atomic partial CSV and resume from completed replicate
groups when the same command is rerun.

## Optional wrapper

The commands below can be run directly or through the documented convenience
wrapper `experiments/reproduction/submit_experiment.sh`. The wrapper executes
the same commands shown in this README. For example:

```bash
./experiments/reproduction/submit_experiment.sh figure1 16 "$OUTPUT_ROOT"
./experiments/reproduction/submit_experiment.sh table4_1 16 "$OUTPUT_ROOT"
./experiments/reproduction/submit_experiment.sh figures3_4_9 16 "$OUTPUT_ROOT"
```

Run the wrapper with `--help` to list every display group. The `all` option
runs the complete workflow and may take more than eight hours. If
`OUTPUT_ROOT` is omitted, the wrapper uses `experiments/generated/`.

## Main-paper experiments

### Figure 1: tail diagnostics

```bash
python -m experiments.reproduction.tail_diagnostics_figure all \
  --output-dir "$OUTPUT_ROOT/tail_diagnostics"
```

Output: `tail_diagnostics.png`, the four component plots, and the numerical
diagnostic arrays under `$OUTPUT_ROOT/tail_diagnostics/`.

### Table 4.1: threshold sensitivity

```bash
python -m experiments.reproduction.synthetic_threshold_sensitivity all \
  --repetitions 200 --estimand both --distribution all --threshold all \
  --workers 16 \
  --output-dir "$OUTPUT_ROOT/synthetic_threshold_sensitivity"
```

Output: replicate-level and aggregate CSV files plus
`tail_probability_rows.tex` for Table 4.1(a) and `quantile_rows.tex` for
Table 4.1(b), under `$OUTPUT_ROOT/synthetic_threshold_sensitivity/`.

### Figure 2 and Table I.1: bounded-support sensitivity

```bash
python -m experiments.reproduction.bounded_support all --workers 16 \
  --output-dir "$OUTPUT_ROOT/bounded_support"
```

Output: `bounded_support.png`, `table_i_1_rows.tex`, and the raw simulation
CSV under `$OUTPUT_ROOT/bounded_support/`.

### Tables 4.2 and 4.3: DRO and EVT comparison

```bash
python -m experiments.reproduction.synthetic_benchmark_comparison all \
  --study all --workers 16 \
  --output-dir "$OUTPUT_ROOT/synthetic_benchmark_comparison"
```

Output: `table_4_2_rows.tex`, `table_4_3_rows.tex`, and the raw method
intervals under `$OUTPUT_ROOT/synthetic_benchmark_comparison/`.

### Figures 3, 4, and 9: GCMT sensitivity analysis

```bash
python -m experiments.reproduction.gcmt_sensitivity_figures all \
  --workers 16 --output-dir "$OUTPUT_ROOT/gcmt_sensitivity_figures"
```

Output: the following plots and the underlying interval CSV under
`$OUTPUT_ROOT/gcmt_sensitivity_figures/`:

- `real_data_vs_evt_threshold_percentage.png` (Figure 3);
- `real_data_vs_evt_critical_values.png` (Figure 4); and
- `real_data_vs_evt_confidence_levels.png` (Figure 9).

### Figure 5: GCMT bootstrap study

```bash
python -m experiments.reproduction.gcmt_bootstrap_figure all --workers 16 \
  --output-dir "$OUTPUT_ROOT/gcmt_bootstrap_figure"
```

Output: `real_data_vs_evt_bootstrap_ecuador_coverage.png` and the
replicate-level and aggregate CSV files under
`$OUTPUT_ROOT/gcmt_bootstrap_figure/`.

## Appendix experiments

### Table F.1(a): tail probability across shape constraints

```bash
python -m experiments.reproduction.synthetic_constraint_comparison all \
  --workers 16 --output-dir "$OUTPUT_ROOT/synthetic_constraint_comparison"
```

Output: `table_f_1a_rows.tex` and the raw and aggregate CSV files under
`$OUTPUT_ROOT/synthetic_constraint_comparison/`.

### Table F.1(b): quantile estimation across shape constraints

```bash
python -m experiments.reproduction.synthetic_quantile_constraint_comparison \
  all --workers 16 \
  --output-dir "$OUTPUT_ROOT/synthetic_quantile_constraint_comparison"
```

Output: `table_f_1b_rows.tex` and the raw and aggregate CSV files under
`$OUTPUT_ROOT/synthetic_quantile_constraint_comparison/`.

### Table F.2: objective-location sensitivity

```bash
python -m experiments.reproduction.synthetic_objective_sensitivity all \
  --workers 16 --output-dir "$OUTPUT_ROOT/synthetic_objective_sensitivity"
```

Output: `table_f_2_rows.tex` and the raw and aggregate CSV files under
`$OUTPUT_ROOT/synthetic_objective_sensitivity/`.

### Tables G.1 and G.3: GCMT DRO tables

```bash
python -m experiments.reproduction.gcmt_dro_tables all --workers 16 \
  --output-dir "$OUTPUT_ROOT/gcmt_dro_tables"
```

Output: `table_g_1_rows.tex`, `table_g_3_rows.tex`, and the raw interval CSV
under `$OUTPUT_ROOT/gcmt_dro_tables/`.

### Table G.2: GCMT descriptive statistics

```bash
python -m experiments.reproduction.gcmt_descriptive_summary all \
  --output-dir "$OUTPUT_ROOT/gcmt_descriptive_summary"
```

Output: `table_rows.tex`, `summary.csv`, and eight regional box plots under
`$OUTPUT_ROOT/gcmt_descriptive_summary/`.

## Reproduction metadata

`manifest.json` maps stable study identifiers to the current manuscript
display numbers, LaTeX labels, and generated artifacts.
The paper also provides the analytical Table B.1 and Figures 6--8; no
experiment command is required for these displays.
