"""End-to-end reproduction pipeline for synthetic threshold sensitivity.

The pipeline has three explicit stages:

1. generate deterministic synthetic samples and solve the DRO problems;
2. aggregate the replicate-level estimates using the manuscript definitions;
3. render the two subtables as LaTeX and compare their cells with jasa_manu.tex.

Run from the repository root with the ``rs`` environment active:

    python -m experiments.reproduction.synthetic_threshold_sensitivity all

The default 200-repetition run is computationally expensive. Use
``--repetitions 1`` only as a smoke test; it cannot reproduce the manuscript
Monte Carlo summaries.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import multiprocessing
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# The project environment includes R under its own prefix. Setting R_HOME
# before importing droevt follows the convention used by the existing
# experiment entry points and allows rpy2 to initialize when the environment's
# Python is invoked by absolute path rather than through ``conda activate``.
ENVIRONMENT_PREFIX = Path(sys.executable).resolve().parent.parent
os.environ.setdefault("R_HOME", str(ENVIRONMENT_PREFIX / "lib" / "R"))
# These synthetic tables predate the May 2025 NumPy bootstrap refactor.
os.environ.setdefault("ROBUSTTAIL_BOOTSTRAP_RNG", "python")

import numpy as np
import pandas as pd
from scipy.stats import gamma, lognorm, pareto

import droevt.utils.synthetic_data_generator as data_utils
from experiments.run_scripts.quantile_estimation.quantileEstimationUnit import (
    quantileEstimationBinarySearchUnit,
)
from experiments.run_scripts.tail_probability.tail_probability_estimation import (
    estimate_tail_probability_D2_chi2_only,
)

LOGGER = logging.getLogger(__name__)
# ``ks::kdde`` may emit the same small-bandwidth/grid warning thousands of
# times during bootstrap calibration. Its meaning is documented in the study
# README; suppressing the duplicate rpy2 console relay does not alter R state
# or computation. Solver-status warnings use a different logger and remain
# visible.
logging.getLogger("rpy2.rinterface_lib.callbacks").setLevel(logging.ERROR)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = (
    REPOSITORY_ROOT
    / "experiments"
    / "generated"
    / "synthetic_threshold_sensitivity"
)
DEFAULT_MANUSCRIPT = REPOSITORY_ROOT.parent / "latex" / "jasa_manu.tex"

RANDOM_SEED = 20220222
DATA_SIZE = 500
ALPHA = 0.05
BOOTSTRAPPING_SIZE = 500
ELLIPSOIDAL_DIMENSION = 3
TAIL_LHS_QUANTILE = 0.99
TAIL_RHS_QUANTILE = 0.995
QUANTILE_LEVEL = 0.99
MONTE_CARLO_Z = 1.96
DEFAULT_WORKERS = min(16, os.cpu_count() or 1)
MAX_TASKS_PER_WORKER = 25

DISTRIBUTIONS = {
    "gamma": gamma,
    "lognorm": lognorm,
    "pareto": pareto,
}

DISPLAY_NAMES = {
    "gamma": "Gamma",
    "lognorm": "Lognorm",
    "pareto": "Pareto",
}

THRESHOLDS: tuple[tuple[str, float | list[float]], ...] = (
    ("60^{th}", 0.60),
    ("70^{th}", 0.70),
    ("80^{th}", 0.80),
    ("90^{th}", 0.90),
    ("multi", [0.60, 0.70, 0.80, 0.90]),
)

RAW_COLUMNS = (
    "estimand",
    "distribution",
    "threshold_label",
    "threshold_spec",
    "repetition",
    "random_seed",
    "estimate",
    "true_value",
    "covered",
)


@dataclass(frozen=True)
class ExperimentTask:
    estimand: str
    distribution: str
    threshold_label: str
    threshold_spec: float | list[float]
    repetition: int

    @property
    def random_seed(self) -> int:
        return RANDOM_SEED + self.repetition


def _tail_probability_estimate(
    distribution: str,
    threshold_spec: float | list[float],
    random_seed: int,
) -> tuple[float, float]:
    data_module = DISTRIBUTIONS[distribution]
    parameters = data_utils.DISTRIBUTION_DEFAULT_PARAMETERS[distribution]
    input_data = data_utils.generate_synthetic_data(
        data_module,
        parameters,
        DATA_SIZE,
        random_seed,
    )
    left_endpoint = data_utils.get_quantile(
        data_module,
        TAIL_LHS_QUANTILE,
        parameters,
    )
    right_endpoint = data_utils.get_quantile(
        data_module,
        TAIL_RHS_QUANTILE,
        parameters,
    )
    estimates = estimate_tail_probability_D2_chi2_only(
        input_data=input_data,
        left_end_point_objective=left_endpoint,
        right_end_point_objective=right_endpoint,
        threshold_percentage=threshold_spec,
        g_ellipsoidal_dimension=ELLIPSOIDAL_DIMENSION,
        alpha=ALPHA,
        random_state=random_seed,
        bootstrapping_size=BOOTSTRAPPING_SIZE,
        right_endpoint=np.inf,
    )
    # The helper returns [lower bound, upper bound]. Table 4.1(a) reports the
    # upper bound under the D=2 ellipsoidal (chi-square) constraint.
    return float(estimates[1]), TAIL_RHS_QUANTILE - TAIL_LHS_QUANTILE


def _quantile_estimate(
    distribution: str,
    threshold_spec: float | list[float],
    random_seed: int,
) -> tuple[float, float]:
    data_module = DISTRIBUTIONS[distribution]
    parameters = data_utils.DISTRIBUTION_DEFAULT_PARAMETERS[distribution]
    input_data = data_utils.generate_synthetic_data(
        data_module,
        parameters,
        DATA_SIZE,
        random_seed,
    )
    true_value = data_utils.get_quantile(
        data_module,
        QUANTILE_LEVEL,
        parameters,
    )
    estimate = quantileEstimationBinarySearchUnit(
        D=2,
        inputData=input_data,
        thresholdPercentage=threshold_spec,
        quantitleValue=QUANTILE_LEVEL,
        gEllipsoidalDimension=ELLIPSOIDAL_DIMENSION,
        alpha=ALPHA,
        random_state=7 * random_seed + 1,
    )
    # ``quantileEstimationPerRep`` calls this established binary-search helper
    # three times for D=0, 1, and 2. Table 4.1(b) reports only its third (D=2)
    # result, so call that same helper directly and avoid two unused solves.
    return float(estimate), float(true_value)


def _run_task(task: ExperimentTask) -> dict[str, object]:
    if task.estimand == "tail_probability":
        estimate, true_value = _tail_probability_estimate(
            task.distribution,
            task.threshold_spec,
            task.random_seed,
        )
    elif task.estimand == "quantile":
        estimate, true_value = _quantile_estimate(
            task.distribution,
            task.threshold_spec,
            task.random_seed,
        )
    else:
        raise ValueError(f"Unknown estimand: {task.estimand}")

    return {
        "estimand": task.estimand,
        "distribution": task.distribution,
        "threshold_label": task.threshold_label,
        "threshold_spec": json.dumps(task.threshold_spec, separators=(",", ":")),
        "repetition": task.repetition,
        "random_seed": task.random_seed,
        "estimate": estimate,
        "true_value": true_value,
        "covered": estimate >= true_value,
    }


def _format_duration(seconds: float) -> str:
    seconds = max(0, round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {seconds:02d}s"
    return f"{minutes:d}m {seconds:02d}s"


def _collect_with_progress(
    results: Iterable[dict[str, object]],
    total: int,
    checkpoint_path: Path,
    initial_rows: Iterable[dict[str, object]] = (),
) -> list[dict[str, object]]:
    rows = list(initial_rows)
    completed_before_run = len(rows)
    remaining_at_start = total - completed_before_run
    started = time.monotonic()
    log_interval = max(1, total // 100)
    for completed_this_run, row in enumerate(results, start=1):
        rows.append(row)
        completed = completed_before_run + completed_this_run
        if (
            completed_this_run == 1
            or completed % log_interval == 0
            or completed == total
        ):
            elapsed = time.monotonic() - started
            eta = (
                elapsed
                / completed_this_run
                * (remaining_at_start - completed_this_run)
            )
            LOGGER.info(
                "Progress %d/%d (%.1f%%); elapsed %s; ETA %s",
                completed,
                total,
                100.0 * completed / total,
                _format_duration(elapsed),
                _format_duration(eta),
            )
            temporary_path = checkpoint_path.with_suffix(
                checkpoint_path.suffix + ".tmp"
            )
            pd.DataFrame(rows, columns=RAW_COLUMNS).to_csv(
                temporary_path,
                index=False,
                float_format="%.17g",
            )
            temporary_path.replace(checkpoint_path)
    return rows


def _task_key(task: ExperimentTask) -> tuple[str, str, str, int]:
    return (
        task.estimand,
        task.distribution,
        task.threshold_label,
        task.repetition,
    )


def _result_key(row: dict[str, object]) -> tuple[str, str, str, int]:
    return (
        str(row["estimand"]),
        str(row["distribution"]),
        str(row["threshold_label"]),
        int(row["repetition"]),
    )


def generate_raw_results(
    output_dir: Path,
    repetitions: int,
    estimands: Iterable[str],
    distributions: Iterable[str],
    thresholds: Iterable[tuple[str, float | list[float]]],
    workers: int,
) -> Path:
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    if workers < 1:
        raise ValueError("workers must be positive")

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "raw_results.partial.csv"
    tasks = [
        ExperimentTask(
            estimand=estimand,
            distribution=distribution,
            threshold_label=threshold_label,
            threshold_spec=threshold_spec,
            repetition=repetition,
        )
        for estimand in estimands
        for distribution in distributions
        for threshold_label, threshold_spec in thresholds
        for repetition in range(repetitions)
    ]
    task_order = {_task_key(task): index for index, task in enumerate(tasks)}
    initial_rows: list[dict[str, object]] = []
    if checkpoint_path.exists():
        checkpoint = pd.read_csv(checkpoint_path)
        missing = set(RAW_COLUMNS) - set(checkpoint.columns)
        if missing:
            raise ValueError(
                f"Checkpoint is missing columns: {sorted(missing)}"
            )
        initial_rows = checkpoint.loc[:, RAW_COLUMNS].to_dict(orient="records")
        checkpoint_keys = {_result_key(row) for row in initial_rows}
        unexpected_keys = checkpoint_keys - set(task_order)
        if unexpected_keys:
            raise ValueError(
                "Checkpoint contains tasks outside the requested run; "
                "use a different output directory"
            )
        if len(checkpoint_keys) != len(initial_rows):
            raise ValueError("Checkpoint contains duplicate experiment tasks")
        tasks = [task for task in tasks if _task_key(task) not in checkpoint_keys]
        LOGGER.info(
            "Resuming from %d checkpointed tasks; %d remain",
            len(initial_rows),
            len(tasks),
        )

    total_tasks = len(task_order)
    LOGGER.info(
        "Running %d remaining task(s) of %d with %d worker(s)",
        len(tasks),
        total_tasks,
        workers,
    )
    if workers == 1:
        rows = _collect_with_progress(
            (_run_task(task) for task in tasks),
            total_tasks,
            checkpoint_path,
            initial_rows,
        )
    else:
        # Spawn gives every worker a clean Python/R/Mosek process. ``imap``
        # yields in input order, so the raw CSV is deterministic regardless of
        # worker count or task completion order while still exposing progress.
        context = multiprocessing.get_context("spawn")
        # Recycling processes bounds growth in the combined R/Mosek worker
        # state during multi-thousand-task runs.
        with context.Pool(
            processes=workers,
            maxtasksperchild=MAX_TASKS_PER_WORKER,
        ) as pool:
            rows = _collect_with_progress(
                pool.imap(_run_task, tasks, chunksize=1),
                total_tasks,
                checkpoint_path,
                initial_rows,
            )

    rows.sort(key=lambda row: task_order[_result_key(row)])
    raw = pd.DataFrame(rows, columns=RAW_COLUMNS)
    raw_path = output_dir / "raw_results.csv"
    raw.to_csv(raw_path, index=False, float_format="%.17g")
    checkpoint_path.unlink(missing_ok=True)
    return raw_path


def aggregate_results(raw_path: Path, output_dir: Path) -> Path:
    raw = pd.read_csv(raw_path)
    missing = set(RAW_COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f"Raw results are missing columns: {sorted(missing)}")

    group_columns = ["estimand", "distribution", "threshold_label"]
    grouped = raw.groupby(group_columns, sort=False, dropna=False)
    summary = grouped.agg(
        repetitions=("estimate", "size"),
        true_value=("true_value", "first"),
        estimate_mean=("estimate", "mean"),
        estimate_std=("estimate", "std"),
        coverage_mean=("covered", "mean"),
        coverage_std=("covered", "std"),
    ).reset_index()

    sqrt_n = np.sqrt(summary["repetitions"])
    summary["estimate_margin"] = (
        MONTE_CARLO_Z * summary["estimate_std"].fillna(0.0) / sqrt_n
    )
    summary["coverage_margin"] = (
        MONTE_CARLO_Z * summary["coverage_std"].fillna(0.0) / sqrt_n
    )
    summary["relative_ratio"] = (
        summary["estimate_mean"] / summary["true_value"]
    )
    summary["relative_ratio_margin"] = (
        summary["estimate_margin"] / summary["true_value"]
    )

    summary_path = output_dir / "summary.csv"
    summary.to_csv(summary_path, index=False, float_format="%.17g")
    return summary_path


def _scientific_latex(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError(f"Cannot format non-finite value: {value}")
    mantissa, exponent = f"{value:.2E}".split("E")
    return rf"{mantissa}\times 10^{{{int(exponent)}}}"


def _result_cells(row: pd.Series) -> str:
    ratio = (
        f"${row.relative_ratio:.3f}\\ (\\pm{row.relative_ratio_margin:.3f})$"
    )
    estimate = (
        f"${_scientific_latex(row.estimate_mean)}"
        f"\\ (\\pm{_scientific_latex(row.estimate_margin)})$"
    )
    coverage = (
        f"${row.coverage_mean:.3f}\\ (\\pm{row.coverage_margin:.3f})$"
    )
    return " & ".join((ratio, estimate, coverage))


def _ordered_summary(summary: pd.DataFrame, estimand: str) -> pd.DataFrame:
    subset = summary.loc[summary["estimand"] == estimand].copy()
    distribution_order = {name: index for index, name in enumerate(DISTRIBUTIONS)}
    threshold_order = {name: index for index, (name, _) in enumerate(THRESHOLDS)}
    subset["_distribution_order"] = subset["distribution"].map(distribution_order)
    subset["_threshold_order"] = subset["threshold_label"].map(threshold_order)
    return subset.sort_values(
        ["_distribution_order", "_threshold_order"], kind="stable"
    )


def _render_rows(summary: pd.DataFrame, estimand: str) -> str:
    lines: list[str] = []
    ordered = _ordered_summary(summary, estimand)
    for distribution, distribution_rows in ordered.groupby(
        "distribution", sort=False
    ):
        true_value = float(distribution_rows["true_value"].iloc[0])
        if estimand == "tail_probability":
            multirow_label = DISPLAY_NAMES[distribution]
        else:
            multirow_label = (
                rf"\makecell{{{DISPLAY_NAMES[distribution]} \\ "
                rf"w. true $q_{{0.99}}={true_value:.2f}$}}"
            )

        for row in distribution_rows.itertuples(index=False):
            prefix = (
                rf"\multirow{{5}}{{*}}{{{multirow_label}}}"
                if row.threshold_label == "60^{th}"
                else ""
            )
            threshold = f"${row.threshold_label}$" if row.threshold_label != "multi" else "multi"
            line = f"{prefix}&{threshold} & {_result_cells(pd.Series(row._asdict()))}\\\\"
            if row.threshold_label == "multi":
                line += r"\hline "
            lines.append(line)
    return "\n".join(lines)


def render_latex(summary_path: Path, output_dir: Path) -> tuple[Path, Path]:
    summary = pd.read_csv(summary_path)
    tail_rows = _render_rows(summary, "tail_probability")
    quantile_rows = _render_rows(summary, "quantile")

    tail_path = output_dir / "tail_probability_rows.tex"
    quantile_path = output_dir / "quantile_rows.tex"
    tail_path.write_text(tail_rows + "\n", encoding="utf-8")
    quantile_path.write_text(quantile_rows + "\n", encoding="utf-8")
    return tail_path, quantile_path


def _extract_manuscript_rows(manuscript_text: str, label: str) -> str:
    # The manuscript retains commented-out historical copies of some tables.
    # Select an active label line rather than the first textual occurrence.
    label_pattern = re.compile(
        rf"^[ \t]*\\label\{{{re.escape(label)}\}}[ \t]*$",
        flags=re.MULTILINE,
    )
    label_matches = list(label_pattern.finditer(manuscript_text))
    if len(label_matches) != 1:
        raise ValueError(
            f"Expected exactly one active manuscript label {label!r}; "
            f"found {len(label_matches)}"
        )
    label_position = label_matches[0].start()
    subtable_start = manuscript_text.rfind(r"\begin{subtable}", 0, label_position)
    subtable_end = manuscript_text.index(r"\end{subtable}", label_position)
    return manuscript_text[subtable_start:subtable_end]


def _parse_cells(latex_rows: str) -> list[tuple[str, ...]]:
    records: list[tuple[str, ...]] = []
    for line in latex_rows.splitlines():
        if r"\pm" not in line:
            continue
        normalized = re.sub(r"\s+", "", line)
        records.append(tuple(normalized.split("&")))
    return records


def _decimal_quantum(text: str) -> float:
    decimal_places = len(text.partition(".")[2])
    return 10.0 ** (-decimal_places)


def _displayed_metrics(row: tuple[str, ...]) -> tuple[tuple[float, float], ...]:
    """Return displayed values and their last-digit quanta for one table row."""
    if len(row) < 5:
        raise ValueError(f"Expected at least five LaTeX cells, got: {row!r}")

    plain_pattern = re.compile(
        r"\$(-?\d+(?:\.\d+)?)\\\(\\pm(-?\d+(?:\.\d+)?)\)"
    )
    scientific_pattern = re.compile(
        r"\$(-?\d+(?:\.\d+)?)\\times10\^\{(-?\d+)\}"
        r"\\\(\\pm(-?\d+(?:\.\d+)?)\\times10\^\{(-?\d+)\}\)"
    )

    ratio_match = plain_pattern.search(row[-3])
    estimate_match = scientific_pattern.search(row[-2])
    coverage_match = plain_pattern.search(row[-1])
    if not (ratio_match and estimate_match and coverage_match):
        raise ValueError(f"Could not parse displayed numerical cells: {row!r}")

    ratio, ratio_margin = ratio_match.groups()
    estimate, estimate_exponent, estimate_margin, margin_exponent = (
        estimate_match.groups()
    )
    coverage, coverage_margin = coverage_match.groups()

    def plain_metric(text: str) -> tuple[float, float]:
        return float(text), _decimal_quantum(text)

    def scientific_metric(text: str, exponent: str) -> tuple[float, float]:
        scale = 10.0 ** int(exponent)
        return float(text) * scale, _decimal_quantum(text) * scale

    return (
        plain_metric(ratio),
        plain_metric(ratio_margin),
        scientific_metric(estimate, estimate_exponent),
        scientific_metric(estimate_margin, margin_exponent),
        plain_metric(coverage),
        plain_metric(coverage_margin),
    )


def _row_distance(
    generated: tuple[str, ...],
    expected: tuple[str, ...],
) -> float:
    generated_metrics = _displayed_metrics(generated)
    expected_metrics = _displayed_metrics(expected)
    return sum(
        abs(generated_value - expected_value) / max(generated_quantum, expected_quantum)
        for (generated_value, generated_quantum), (expected_value, expected_quantum)
        in zip(generated_metrics, expected_metrics)
    )


def _within_acceptance_tolerance(
    generated: tuple[str, ...],
    expected: tuple[str, ...],
) -> bool:
    generated_metrics = _displayed_metrics(generated)
    expected_metrics = _displayed_metrics(expected)
    # The substantive tolerance requested for Table 4.1(a) is 1e-4 on the
    # underlying probability estimate. Relative ratios divide that estimate by
    # the true value 0.005. Coverage is empirical over 200 repetitions; a
    # difference of up to two repetitions is accepted for this study.
    tolerances = (
        1e-4 / (TAIL_RHS_QUANTILE - TAIL_LHS_QUANTILE),
        1e-4 / (TAIL_RHS_QUANTILE - TAIL_LHS_QUANTILE),
        1e-4,
        1e-4,
        2.0 / 200.0,
        1e-2,
    )
    return all(
        abs(generated_value - expected_value) <= tolerance * (1.0 + 1e-9)
        for ((generated_value, _), (expected_value, _)), tolerance
        in zip(zip(generated_metrics, expected_metrics), tolerances)
    )


def verify_against_manuscript(
    output_dir: Path,
    manuscript_path: Path,
    require_full_repetitions: bool,
) -> None:
    summary = pd.read_csv(output_dir / "summary.csv")
    if require_full_repetitions and not (summary["repetitions"] == 200).all():
        raise ValueError("Ground-truth verification requires exactly 200 repetitions")

    manuscript_text = manuscript_path.read_text(encoding="utf-8")
    comparisons = (
        ("tb3_tpe", output_dir / "tail_probability_rows.tex"),
        ("tb3_qe", output_dir / "quantile_rows.tex"),
    )
    failures: list[str] = []
    report_rows: list[tuple[str, str, tuple[str, ...], tuple[str, ...]]] = []
    for label, generated_path in comparisons:
        expected = _parse_cells(_extract_manuscript_rows(manuscript_text, label))
        generated = _parse_cells(generated_path.read_text(encoding="utf-8"))
        for generated_row in generated:
            if generated_row in expected:
                report_rows.append(
                    (label, "exact", generated_row, generated_row)
                )
                continue

            # A partial run may omit the distribution multirow prefix. Match on
            # threshold first, then choose the numerically nearest manuscript row.
            candidates = [
                row
                for row in expected
                if len(row) >= 5 and row[1] == generated_row[1]
            ]
            if generated_row[0]:
                candidates = [
                    row for row in candidates if row[0] == generated_row[0]
                ]
            if not candidates:
                failures.append(
                    f"{label}: no manuscript row matches key "
                    f"{generated_row[:2]!r}"
                )
                continue

            expected_row = min(
                candidates,
                key=lambda row: _row_distance(generated_row, row),
            )
            if _within_acceptance_tolerance(generated_row, expected_row):
                status = "accepted numerical deviation"
            else:
                status = "outside tolerance"
                failures.append(
                    f"{label}: generated row is outside the documented numerical "
                    f"acceptance tolerance"
                )
            report_rows.append((label, status, generated_row, expected_row))

    report_lines = [
        "# Manuscript verification report",
        "",
        "Byte-for-byte equality remains the primary target. A numerical row is",
        "marked accepted when the underlying probability estimate and its",
        "Monte Carlo half-width are within `1e-4`. The corresponding ratio",
        "tolerance is `1e-4 / 0.005 = 0.02`. Empirical coverage may differ by",
        "at most two of 200 repetitions (`0.01`), with a `0.01` tolerance for",
        "its displayed half-width.",
        "",
        "| Label | Status | Generated row | Manuscript row |",
        "| --- | --- | --- | --- |",
    ]
    for label, status, generated_row, expected_row in report_rows:
        generated_text = " & ".join(generated_row).replace("|", r"\|")
        expected_text = " & ".join(expected_row).replace("|", r"\|")
        report_lines.append(
            f"| `{label}` | {status} | `{generated_text}` | `{expected_text}` |"
        )
    (output_dir / "verification_report.md").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    if failures:
        raise AssertionError("\n".join(failures))


def _parse_estimands(value: str) -> tuple[str, ...]:
    if value == "both":
        return ("tail_probability", "quantile")
    return (value,)


def _parse_distributions(value: str) -> tuple[str, ...]:
    if value == "all":
        return tuple(DISTRIBUTIONS)
    return (value,)


def _parse_thresholds(
    value: str,
) -> tuple[tuple[str, float | list[float]], ...]:
    if value == "all":
        return THRESHOLDS
    requested = tuple(part.strip() for part in value.split(",") if part.strip())
    valid = {"60", "70", "80", "90", "multi"}
    invalid = set(requested) - valid
    if not requested or invalid:
        raise ValueError(
            "Threshold must be 'all' or a comma-separated subset of "
            f"{sorted(valid)}; invalid values: {sorted(invalid)}"
        )
    labels = {
        "multi" if threshold == "multi" else f"{threshold}^{{th}}"
        for threshold in requested
    }
    return tuple(item for item in THRESHOLDS if item[0] in labels)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=("generate", "aggregate", "render", "verify", "all"),
    )
    parser.add_argument("--repetitions", type=int, default=200)
    parser.add_argument(
        "--estimand",
        choices=("tail_probability", "quantile", "both"),
        default="both",
    )
    parser.add_argument(
        "--distribution",
        choices=(*DISTRIBUTIONS, "all"),
        default="all",
    )
    parser.add_argument(
        "--threshold",
        default="all",
        help=(
            "Threshold selection: 'all' or a comma-separated subset of "
            "60,70,80,90,multi"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manuscript", type=Path, default=DEFAULT_MANUSCRIPT)
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=(
            "Number of spawned worker processes used during generation "
            f"(default: {DEFAULT_WORKERS}; use 1 for sequential debugging)"
        ),
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.stage in {"generate", "all"}:
        generate_raw_results(
            args.output_dir,
            args.repetitions,
            _parse_estimands(args.estimand),
            _parse_distributions(args.distribution),
            _parse_thresholds(args.threshold),
            args.workers,
        )
    if args.stage in {"aggregate", "all"}:
        aggregate_results(args.output_dir / "raw_results.csv", args.output_dir)
    if args.stage in {"render", "all"}:
        render_latex(args.output_dir / "summary.csv", args.output_dir)
    if args.stage == "verify":
        verify_against_manuscript(
            args.output_dir,
            args.manuscript,
            require_full_repetitions=True,
        )


if __name__ == "__main__":
    main()
