# Reproducing the manuscript experiments

The commands below regenerate the experimental tables and figures in the
manuscript. Run them from the repository root. Generated files are written
under `experiments/generated/`.

## Environment

Create and activate the supplied `rs` Conda environment:

```bash
conda env create -f environment_linux.yml
conda activate rs
```

On Apple Silicon, use `environment_osx-arm64.yml` instead. The optimization
experiments require a valid MOSEK license; installation details are in the
repository's main `README.md`.

The simulation runners below use 16 workers. A smaller value can be passed
with `--workers` when fewer CPU cores are available. Long-running simulations
write an atomic partial CSV and resume from completed replicate groups when
the same command is rerun.

## Main-paper experiments

### Figure 1: tail diagnostics

```bash
python -m experiments.reproduction.tail_diagnostics_figure all
```

Output: `experiments/generated/tail_diagnostics/tail_diagnostics.png`, the
four component plots, and the numerical diagnostic arrays.

### Table 4.1: threshold sensitivity

```bash
python -m experiments.reproduction.synthetic_threshold_sensitivity generate \
  --repetitions 200 --estimand both --distribution all --threshold all \
  --workers 16
python -m experiments.reproduction.synthetic_threshold_sensitivity aggregate
python -m experiments.reproduction.synthetic_threshold_sensitivity render
```

Output: replicate-level and aggregate CSV files plus
`tail_probability_rows.tex` for Table 4.1(a) and `quantile_rows.tex` for
Table 4.1(b), under
`experiments/generated/synthetic_threshold_sensitivity/`.

### Figure 2 and Table I.1: bounded-support sensitivity

```bash
python -m experiments.reproduction.bounded_support all --workers 16
```

Output: `bounded_support.png`, `table_i_1_rows.tex`, and the raw simulation
CSV under `experiments/generated/bounded_support/`.

### Tables 4.2 and 4.3: DRO and EVT comparison

```bash
python -m experiments.reproduction.synthetic_benchmark_comparison all \
  --study all --workers 16
```

Output: `table_4_2_rows.tex`, `table_4_3_rows.tex`, and the raw method
intervals under
`experiments/generated/synthetic_benchmark_comparison/`.

### Figures 3, 4, and 9: GCMT sensitivity analysis

```bash
python -m experiments.reproduction.gcmt_sensitivity_figures all \
  --workers 16
```

Output: the following plots and the underlying interval CSV under
`experiments/generated/gcmt_sensitivity_figures/`:

- `real_data_vs_evt_threshold_percentage.png` (Figure 3);
- `real_data_vs_evt_critical_values.png` (Figure 4); and
- `real_data_vs_evt_confidence_levels.png` (Figure 9).

### Figure 5: GCMT bootstrap study

```bash
python -m experiments.reproduction.gcmt_bootstrap_figure all --workers 16
```

Output: `real_data_vs_evt_bootstrap_ecuador_coverage.png` and the
replicate-level and aggregate CSV files under
`experiments/generated/gcmt_bootstrap_figure/`.

## Appendix experiments

### Table F.1(a): tail probability across shape constraints

```bash
python -m experiments.reproduction.synthetic_constraint_comparison all \
  --workers 16
```

Output: `table_f_1a_rows.tex` and the raw and aggregate CSV files under
`experiments/generated/synthetic_constraint_comparison/`.

### Table F.1(b): quantile estimation across shape constraints

```bash
python -m experiments.reproduction.synthetic_quantile_constraint_comparison \
  all --workers 16
```

Output: `table_f_1b_rows.tex` and the raw and aggregate CSV files under
`experiments/generated/synthetic_quantile_constraint_comparison/`.

### Table F.2: objective-location sensitivity

```bash
python -m experiments.reproduction.synthetic_objective_sensitivity all \
  --workers 16
```

Output: `table_f_2_rows.tex` and the raw and aggregate CSV files under
`experiments/generated/synthetic_objective_sensitivity/`.

### Tables G.1 and G.3: GCMT DRO tables

```bash
python -m experiments.reproduction.gcmt_dro_tables all --workers 16
```

Output: `table_g_1_rows.tex`, `table_g_3_rows.tex`, and the raw interval CSV
under
`experiments/generated/gcmt_dro_tables/`.

### Table G.2: GCMT descriptive statistics

```bash
python -m experiments.reproduction.gcmt_descriptive_summary all
```

Output: `table_rows.tex`, `summary.csv`, and eight regional box plots under
`experiments/generated/gcmt_descriptive_summary/`.

## Reproduction metadata

`manifest.json` maps stable study identifiers to the current manuscript
display numbers, LaTeX labels, and generated artifacts.
The analytical Table B.1 and Figures 6--8 are not experiments and therefore
do not require simulation scripts.
