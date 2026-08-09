#!/usr/bin/env bash
set -euo pipefail

# Convenience wrapper for the commands documented in this directory's
# README.md. Run one display group at a time, for example:
#   ./experiments/reproduction/submit_experiment.sh figure1 16 reproduced-results
#   ./experiments/reproduction/submit_experiment.sh table4_1 16 reproduced-results
#
# The optional second argument is the number of workers (default: 16).
# The optional third argument is the output root (default:
# experiments/generated); each display group receives its own subdirectory.
# Activate the supplied `rs` Conda environment before running this script.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd "${script_dir}/../.." && pwd)"
cd "${repository_root}"

study="${1:-help}"
workers="${2:-16}"
output_root="${3:-${repository_root}/experiments/generated}"

run_study() {
    case "$1" in
        figure1)
            python -m experiments.reproduction.tail_diagnostics_figure all \
                --output-dir "${output_root}/tail_diagnostics"
            ;;
        table4_1)
            python -m experiments.reproduction.synthetic_threshold_sensitivity all \
                --repetitions 200 --estimand both --distribution all \
                --threshold all --workers "${workers}" \
                --output-dir "${output_root}/synthetic_threshold_sensitivity"
            ;;
        figure2_table_i1)
            python -m experiments.reproduction.bounded_support all \
                --workers "${workers}" \
                --output-dir "${output_root}/bounded_support"
            ;;
        tables4_2_4_3)
            python -m experiments.reproduction.synthetic_benchmark_comparison all \
                --study all --workers "${workers}" \
                --output-dir "${output_root}/synthetic_benchmark_comparison"
            ;;
        figures3_4_9)
            python -m experiments.reproduction.gcmt_sensitivity_figures all \
                --workers "${workers}" \
                --output-dir "${output_root}/gcmt_sensitivity_figures"
            ;;
        figure5)
            python -m experiments.reproduction.gcmt_bootstrap_figure all \
                --workers "${workers}" \
                --output-dir "${output_root}/gcmt_bootstrap_figure"
            ;;
        table_f1a)
            python -m experiments.reproduction.synthetic_constraint_comparison all \
                --workers "${workers}" \
                --output-dir "${output_root}/synthetic_constraint_comparison"
            ;;
        table_f1b)
            python -m \
                experiments.reproduction.synthetic_quantile_constraint_comparison \
                all --workers "${workers}" \
                --output-dir \
                "${output_root}/synthetic_quantile_constraint_comparison"
            ;;
        table_f2)
            python -m experiments.reproduction.synthetic_objective_sensitivity all \
                --workers "${workers}" \
                --output-dir "${output_root}/synthetic_objective_sensitivity"
            ;;
        tables_g1_g3)
            python -m experiments.reproduction.gcmt_dro_tables all \
                --workers "${workers}" \
                --output-dir "${output_root}/gcmt_dro_tables"
            ;;
        table_g2)
            python -m experiments.reproduction.gcmt_descriptive_summary all \
                --output-dir "${output_root}/gcmt_descriptive_summary"
            ;;
        *)
            echo "Unknown display group: $1" >&2
            return 2
            ;;
    esac
}

display_groups=(
    figure1
    table4_1
    figure2_table_i1
    tables4_2_4_3
    figures3_4_9
    figure5
    table_f1a
    table_f1b
    table_f2
    tables_g1_g3
    table_g2
)

case "${study}" in
    help|-h|--help)
        echo "Usage: $0 DISPLAY_GROUP [WORKERS] [OUTPUT_ROOT]"
        echo
        echo "DISPLAY_GROUP:"
        printf '  %s\n' "${display_groups[@]}"
        echo "  all"
        echo
        echo "WORKERS is the number of parallel Python processes (default: 16)."
        echo "It controls CPU concurrency, not Monte Carlo repetitions."
        echo "OUTPUT_ROOT defaults to experiments/generated."
        ;;
    all)
        for display_group in "${display_groups[@]}"; do
            echo "Running ${display_group}"
            run_study "${display_group}"
        done
        ;;
    *)
        run_study "${study}"
        ;;
esac
