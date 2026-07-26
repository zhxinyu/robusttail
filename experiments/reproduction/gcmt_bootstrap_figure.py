"""Reproduce GCMT bootstrap Figure 5 end to end.

This factors the retained ``exp_cmt.py`` bootstrap logic and fills in the
union of sample-size settings used by its historical and current versions so
that the grid matches the active manuscript: 11 sizes, 8 critical quantiles,
200 repetitions, and five methods.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import multiprocessing
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ENVIRONMENT_PREFIX = Path(sys.executable).resolve().parent.parent
os.environ.setdefault("R_HOME", str(ENVIRONMENT_PREFIX / "lib" / "R"))
os.environ.setdefault("ROBUSTTAIL_BOOTSTRAP_RNG", "numpy")

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import zscore

from experiments.input_data.cmt.parse_script import parse_ndk
from experiments.run_scripts.tail_probability.benchmark_tail_probability_estimation import (
    benchmark_estimate_tail_probability,
)
from experiments.run_scripts.tail_probability.tail_probability_estimation import (
    estimate_tail_probability_D2_chi2_only,
)

LOGGER = logging.getLogger(__name__)
logging.getLogger("rpy2.rinterface_lib.callbacks").setLevel(logging.ERROR)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MANUSCRIPT_FIGURE = (
    REPOSITORY_ROOT.parent
    / "latex"
    / "plots"
    / "real_data_vs_evt_bootstrap_ecuador_coverage.png"
)
DEFAULT_OUTPUT_DIR = (
    REPOSITORY_ROOT / "experiments" / "generated" / "gcmt_bootstrap_figure"
)

REGION = "ECUADOR"
SAMPLE_SIZES = tuple(range(50, 151, 10))
CRITICAL_QUANTILES = (0.990, 0.992, 0.994, 0.996, 0.998, 0.9991, 0.9993, 0.9995)
METHODS = ("dro", "pot_bt", "pl", "bayesian", "pwm")
METHOD_LABELS = ("DRO", "MLE-v2", "PL", "BI", "PWM")
REPETITIONS = 200
RANDOM_SEED = 20220222
THRESHOLD_PERCENTAGE = 0.70
BOOTSTRAPPING_SIZE = 500
ELLIPSOIDAL_DIMENSION = 3
ALPHA = 0.05
DEFAULT_WORKERS = min(16, os.cpu_count() or 1)
MAX_TASKS_PER_WORKER = 25
FIELDS = (
    "sample_size",
    "critical_quantile",
    "repetition",
    "method",
    "lower_bound",
    "upper_bound",
)
SPOT_SETTINGS = (
    (50, 0.9995, 0),
    (100, 0.9960, 11),
    (150, 0.9900, 0),
)


@dataclass(frozen=True)
class Task:
    sample_size: int
    critical_quantile: float
    repetition: int
    sample: np.ndarray
    objective: float

    @property
    def key(self) -> tuple[int, float, int]:
        return self.sample_size, self.critical_quantile, self.repetition


def _full_data() -> np.ndarray:
    frame = parse_ndk()
    values = frame.loc[frame["location"] == REGION, "Mw"].to_numpy(dtype=float)
    return np.asarray(zscore(values), dtype=float)


def _tasks() -> tuple[list[Task], np.ndarray]:
    full = _full_data()
    tasks: list[Task] = []
    # exp_cmt.py resets the seed for every critical quantile.  Consequently,
    # each quantile uses the same 200 bootstrap index sequences for a fixed n.
    for sample_size in SAMPLE_SIZES:
        samples_by_repetition: list[np.ndarray] = []
        rng = np.random.RandomState(RANDOM_SEED)
        for _ in range(REPETITIONS):
            samples_by_repetition.append(
                np.asarray(rng.choice(full, size=sample_size, replace=True), dtype=float)
            )
        for quantile in CRITICAL_QUANTILES:
            objective = float(np.quantile(full, quantile))
            tasks.extend(
                Task(sample_size, quantile, repetition, sample, objective)
                for repetition, sample in enumerate(samples_by_repetition)
            )
    return tasks, full


def _spot_tasks() -> list[Task]:
    """Build representative corner/interior tasks without materializing the full grid."""
    full = _full_data()
    tasks: list[Task] = []
    for sample_size, quantile, repetition in SPOT_SETTINGS:
        rng = np.random.RandomState(RANDOM_SEED)
        sample = np.empty(0, dtype=float)
        for _ in range(repetition + 1):
            sample = np.asarray(
                rng.choice(full, size=sample_size, replace=True), dtype=float
            )
        tasks.append(
            Task(
                sample_size,
                quantile,
                repetition,
                sample,
                float(np.quantile(full, quantile)),
            )
        )
    return tasks


def _interval(result: object) -> tuple[float, float]:
    try:
        values = list(result)  # type: ignore[arg-type]
        if len(values) != 2:
            return 0.0, 0.0
        lower, upper = float(values[0]), float(values[1])
        if not math.isfinite(lower) or not math.isfinite(upper):
            return math.nan, math.nan
        return lower, upper
    except (TypeError, ValueError):
        return 0.0, 0.0


def _run_task(task: Task) -> list[dict[str, object]]:
    intervals: dict[str, tuple[float, float]] = {}
    intervals["dro"] = _interval(
        estimate_tail_probability_D2_chi2_only(
            input_data=task.sample,
            left_end_point_objective=task.objective,
            right_end_point_objective=np.inf,
            threshold_percentage=THRESHOLD_PERCENTAGE,
            g_ellipsoidal_dimension=ELLIPSOIDAL_DIMENSION,
            alpha=ALPHA,
            random_state=RANDOM_SEED,
            bootstrapping_size=BOOTSTRAPPING_SIZE,
            right_endpoint=np.inf,
        )
    )
    for method in METHODS[1:]:
        intervals[method] = _interval(
            benchmark_estimate_tail_probability(
                input_data=task.sample,
                left_end_point_objective=task.objective,
                right_end_point_objective=np.inf,
                method=method,
                alpha=ALPHA,
                random_state=RANDOM_SEED + task.repetition,
            )
        )
    return [
        {
            "sample_size": task.sample_size,
            "critical_quantile": task.critical_quantile,
            "repetition": task.repetition,
            "method": method,
            "lower_bound": intervals[method][0],
            "upper_bound": intervals[method][1],
        }
        for method in METHODS
    ]


def _read(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_atomic(path: Path, rows: list[dict[str, object]], order: dict[tuple, int]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    ordered = sorted(
        rows,
        key=lambda row: order[
            (
                int(row["sample_size"]),
                float(row["critical_quantile"]),
                int(row["repetition"]),
                str(row["method"]),
            )
        ],
    )
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(ordered)
    temporary.replace(path)


def generate(output_dir: Path, workers: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks, _ = _tasks()
    row_keys = [
        (task.sample_size, task.critical_quantile, task.repetition, method)
        for task in tasks
        for method in METHODS
    ]
    order = {key: index for index, key in enumerate(row_keys)}
    final_path = output_dir / "raw_results.csv"
    partial_path = output_dir / "raw_results.partial.csv"
    existing = _read(final_path) or _read(partial_path)
    rows: list[dict[str, object]] = [dict(row) for row in existing]
    completed_methods: dict[tuple[int, float, int], set[str]] = {}
    for row in existing:
        key = (
            int(row["sample_size"]),
            float(row["critical_quantile"]),
            int(row["repetition"]),
        )
        completed_methods.setdefault(key, set()).add(row["method"])
    malformed = [key for key, methods in completed_methods.items() if methods != set(METHODS)]
    if malformed:
        raise ValueError(f"Checkpoint has incomplete task groups: {malformed[:3]}")
    completed = set(completed_methods)
    expected = {task.key for task in tasks}
    if not completed.issubset(expected):
        raise ValueError("Output directory contains rows outside the manuscript grid")
    pending = [task for task in tasks if task.key not in completed]
    LOGGER.info(
        "%d/%d bootstrap groups complete (%d method rows); %d pending",
        len(completed),
        len(tasks),
        len(existing),
        len(pending),
    )
    started = time.monotonic()
    context = multiprocessing.get_context("spawn")
    with context.Pool(
        processes=workers,
        maxtasksperchild=MAX_TASKS_PER_WORKER,
    ) as pool:
        for completed_now, result_rows in enumerate(
            pool.imap_unordered(_run_task, pending, chunksize=1), start=1
        ):
            rows.extend(result_rows)
            done = len(completed) + completed_now
            if completed_now == 1 or done % max(1, len(tasks) // 100) == 0 or done == len(tasks):
                _write_atomic(partial_path, rows, order)
                elapsed = time.monotonic() - started
                rate = completed_now / max(elapsed, 1e-9)
                LOGGER.info(
                    "Progress %d/%d groups (%.1f%%); elapsed %.1fs; ETA %.1fs",
                    done,
                    len(tasks),
                    100 * done / len(tasks),
                    elapsed,
                    (len(tasks) - done) / rate,
                )
    _write_atomic(final_path, rows, order)
    partial_path.unlink(missing_ok=True)
    return final_path


def aggregate(raw_path: Path) -> list[dict[str, object]]:
    rows = _read(raw_path)
    expected_count = (
        len(SAMPLE_SIZES)
        * len(CRITICAL_QUANTILES)
        * REPETITIONS
        * len(METHODS)
    )
    if len(rows) != expected_count:
        raise ValueError(f"Raw results incomplete: {len(rows)}/{expected_count}")
    grouped: dict[tuple[int, float, str], list[tuple[float, float]]] = {}
    for row in rows:
        key = int(row["sample_size"]), float(row["critical_quantile"]), row["method"]
        grouped.setdefault(key, []).append(
            (float(row["lower_bound"]), float(row["upper_bound"]))
        )
    result: list[dict[str, object]] = []
    for sample_size in SAMPLE_SIZES:
        for quantile in CRITICAL_QUANTILES:
            # The manuscript figure evaluates coverage for the exceedance
            # probability associated with the named quantile level, 1-q.
            # Using the discrete catalog fraction above the interpolated
            # empirical quantile would equal 1/67 throughout this upper-tail
            # grid and gives the opposite column trend from the stored figure.
            true_probability = 1.0 - quantile
            for method in METHODS:
                values = np.asarray(grouped[(sample_size, quantile, method)], dtype=float)
                if values.shape != (REPETITIONS, 2):
                    raise ValueError((sample_size, quantile, method, values.shape))
                finite = np.isfinite(values).all(axis=1)
                if np.any(values[finite, 0] > values[finite, 1]):
                    raise ValueError(
                        f"Reversed interval for {(sample_size, quantile, method)}"
                    )
                coverage = (
                    finite
                    & (values[:, 0] <= true_probability)
                    & (true_probability <= values[:, 1])
                )
                widths = values[finite, 1] - values[finite, 0]
                result.append(
                    {
                        "sample_size": sample_size,
                        "critical_quantile": quantile,
                        "method": method,
                        "true_probability": true_probability,
                        "coverage": float(np.mean(coverage)),
                        "mean_width": float(np.mean(widths)),
                        "missing_intervals": int((~finite).sum()),
                    }
                )
    return result


def write_aggregate(raw_path: Path, output_dir: Path) -> Path:
    rows = aggregate(raw_path)
    path = output_dir / "aggregate_results.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return path


def plot(raw_path: Path, output_dir: Path) -> Path:
    rows = aggregate(raw_path)
    lookup = {
        (int(row["sample_size"]), float(row["critical_quantile"]), str(row["method"])): row
        for row in rows
    }
    displayed_sizes = tuple(reversed(SAMPLE_SIZES))
    coverage_grid = np.asarray(
        [
            [
                float(lookup[(size, quantile, "dro")]["coverage"])
                for quantile in CRITICAL_QUANTILES
            ]
            for size in displayed_sizes
        ]
    )
    figure, axis = plt.subplots(figsize=(24, 13))
    norm = colors.TwoSlopeNorm(vmin=0.85, vcenter=0.95, vmax=1.0)
    image = axis.imshow(coverage_grid, cmap="RdYlGn", norm=norm, aspect="auto")
    axis.set_xticks(
        np.arange(len(CRITICAL_QUANTILES)),
        [f"{value:.4f}" for value in CRITICAL_QUANTILES],
        fontsize=13,
    )
    axis.set_yticks(
        np.arange(len(displayed_sizes)),
        [str(value) for value in displayed_sizes],
        fontsize=13,
    )
    axis.set_xlabel("Critical quantile", fontsize=15)
    axis.set_ylabel("Bootstrap sample size", fontsize=15)
    for row_index, sample_size in enumerate(displayed_sizes):
        for column_index, quantile in enumerate(CRITICAL_QUANTILES):
            selected = [lookup[(sample_size, quantile, method)] for method in METHODS]
            coverages = [float(row["coverage"]) for row in selected]
            widths = [float(row["mean_width"]) for row in selected]
            best = int(np.argmax(coverages))
            top_parts = []
            for method_index, value in enumerate(coverages):
                top_parts.append(
                    rf"$\mathbf{{{value:.2f}}}$" if method_index == best else f"{value:.2f}"
                )
            axis.text(
                column_index,
                row_index - 0.16,
                " ".join(top_parts),
                ha="center",
                va="center",
                fontsize=10,
            )
            axis.text(
                column_index,
                row_index + 0.18,
                " ".join(f"{value:.2E}" for value in widths),
                ha="center",
                va="center",
                fontsize=6.5,
            )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.025, pad=0.005)
    colorbar.set_label("DRO empirical coverage", fontsize=13)
    figure.tight_layout()
    path = output_dir / MANUSCRIPT_FIGURE.name
    figure.savefig(path, dpi=300)
    plt.close(figure)
    return path


def verify(raw_path: Path, output_dir: Path) -> Path:
    rows = aggregate(raw_path)
    expected_aggregates = len(SAMPLE_SIZES) * len(CRITICAL_QUANTILES) * len(METHODS)
    if len(rows) != expected_aggregates:
        raise ValueError((len(rows), expected_aggregates))
    generated = output_dir / MANUSCRIPT_FIGURE.name
    if not generated.exists() or not MANUSCRIPT_FIGURE.exists():
        raise FileNotFoundError(generated)
    dro_coverages = [
        float(row["coverage"]) for row in rows if row["method"] == "dro"
    ]
    method_summaries = {}
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        coverages = np.asarray([float(row["coverage"]) for row in selected])
        widths = np.asarray([float(row["mean_width"]) for row in selected])
        method_summaries[method] = {
            "coverage_min": float(np.min(coverages)),
            "coverage_mean": float(np.mean(coverages)),
            "coverage_max": float(np.max(coverages)),
            "width_mean": float(np.mean(widths)),
        }
    dro_at_smallest_n = np.mean(
        [
            float(row["coverage"])
            for row in rows
            if row["method"] == "dro" and row["sample_size"] == min(SAMPLE_SIZES)
        ]
    )
    dro_at_largest_n = np.mean(
        [
            float(row["coverage"])
            for row in rows
            if row["method"] == "dro" and row["sample_size"] == max(SAMPLE_SIZES)
        ]
    )
    dro_at_lowest_q = np.mean(
        [
            float(row["coverage"])
            for row in rows
            if row["method"] == "dro"
            and row["critical_quantile"] == min(CRITICAL_QUANTILES)
        ]
    )
    dro_at_highest_q = np.mean(
        [
            float(row["coverage"])
            for row in rows
            if row["method"] == "dro"
            and row["critical_quantile"] == max(CRITICAL_QUANTILES)
        ]
    )
    missing = sum(int(row["missing_intervals"]) for row in rows)
    generated_shape = plt.imread(generated).shape
    expected_shape = plt.imread(MANUSCRIPT_FIGURE).shape
    report = [
        "# Figure 5 verification report",
        "",
        f"- Bootstrap groups: {len(SAMPLE_SIZES) * len(CRITICAL_QUANTILES) * REPETITIONS}",
        f"- Raw method intervals: {len(SAMPLE_SIZES) * len(CRITICAL_QUANTILES) * REPETITIONS * len(METHODS)}",
        f"- Aggregate cells: {len(rows)}/{expected_aggregates}",
        f"- Missing/non-finite method intervals retained as coverage failures: {missing}",
        f"- DRO coverage range: {min(dro_coverages):.3f} to {max(dro_coverages):.3f}",
        f"- Mean DRO coverage at sample size 50 versus 150: {dro_at_smallest_n:.3f} versus {dro_at_largest_n:.3f}",
        f"- Mean DRO coverage at quantiles 0.9900 versus 0.9995: {dro_at_lowest_q:.3f} versus {dro_at_highest_q:.3f}",
        "- Existing implementations reused: D=2 chi-square DRO, MLE-v2, PL, BI, and PWM",
        "- Repetitions: 200 per `(sample size, critical quantile)` setting",
        "- R-based estimators use the explicit `20220222 + repetition` seed, making BI MCMC repeatable",
        "- Coverage target: the manuscript quantile-level exceedance probability `1 - q`",
        "- Plot criterion: substantive scientific and visual equivalence; PNG byte equality is not required",
        f"- Generated pixels: {generated_shape[1]}×{generated_shape[0]}",
        f"- Manuscript pixels: {expected_shape[1]}×{expected_shape[0]}",
        "",
        "| Method | Minimum coverage | Mean coverage | Maximum coverage | Mean interval width |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for method, label in zip(METHODS, METHOD_LABELS):
        summary = method_summaries[method]
        report.append(
            f"| {label} | {summary['coverage_min']:.3f} | "
            f"{summary['coverage_mean']:.3f} | {summary['coverage_max']:.3f} | "
            f"{summary['width_mean']:.6g} |"
        )
    report.extend(
        [
        "",
        "Acceptance requires the same 11-by-8 grid, method ordering, displayed coverage and width patterns, DRO shading, and scientific conclusions. Visual review is still required before acceptance.",
        ]
    )
    path = output_dir / "verification_report.md"
    path.write_text("\n".join(report) + "\n", encoding="utf-8")
    return path


def spot_check(output_dir: Path, workers: int) -> Path:
    """Audit representative raw intervals and deterministic seeded replay."""
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = _spot_tasks()
    repeated_tasks = tasks + tasks
    LOGGER.info(
        "Spot checking %d settings twice with %d workers",
        len(tasks),
        min(workers, len(repeated_tasks)),
    )
    context = multiprocessing.get_context("spawn")
    with context.Pool(
        processes=min(workers, len(repeated_tasks)),
        maxtasksperchild=1,
    ) as pool:
        results = list(pool.imap(_run_task, repeated_tasks, chunksize=1))
    first, replay = results[: len(tasks)], results[len(tasks) :]
    rows = [row for group in first for row in group]
    raw_path = output_dir / "spot_check_results.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    replay_checks: list[dict[str, object]] = []
    for task, original_group, replay_group in zip(tasks, first, replay):
        for original, repeated in zip(original_group, replay_group):
            difference = max(
                abs(float(original[field]) - float(repeated[field]))
                for field in ("lower_bound", "upper_bound")
            )
            replay_checks.append(
                {
                    "sample_size": task.sample_size,
                    "critical_quantile": task.critical_quantile,
                    "repetition": task.repetition,
                    "method": original["method"],
                    "maximum_absolute_difference": difference,
                }
            )
    replay_path = output_dir / "spot_check_replay_comparisons.csv"
    with replay_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=replay_checks[0].keys())
        writer.writeheader()
        writer.writerows(replay_checks)

    finite = sum(
        math.isfinite(float(row["lower_bound"]))
        and math.isfinite(float(row["upper_bound"]))
        for row in rows
    )
    ordered = sum(
        float(row["lower_bound"]) <= float(row["upper_bound"]) for row in rows
    )
    zero_fallbacks = sum(
        float(row["lower_bound"]) == 0.0 and float(row["upper_bound"]) == 0.0
        for row in rows
    )
    target_contained = sum(
        float(row["lower_bound"])
        <= 1.0 - float(row["critical_quantile"])
        <= float(row["upper_bound"])
        for row in rows
    )
    exact_replays = sum(
        float(item["maximum_absolute_difference"]) <= 1e-12
        for item in replay_checks
    )
    report = [
        "# Figure 5 raw-interval spot check",
        "",
        f"- Representative `(sample size, critical quantile, repetition)` settings: {len(tasks)}",
        f"- Methods per setting: {len(METHODS)}",
        f"- Finite intervals: {finite}/{len(rows)}",
        f"- Ordered intervals: {ordered}/{len(rows)}",
        f"- Explicit `[0, 0]` estimator fallbacks: {zero_fallbacks}/{len(rows)}",
        f"- Selected target probabilities contained: {target_contained}/{len(rows)}",
        f"- Seeded replay matches within `1e-12`: {exact_replays}/{len(replay_checks)}",
        "- Settings cover the smallest/hardest corner, an interior point, and the largest/easiest corner of the manuscript grid",
        "- Existing D=2 chi-square DRO, MLE-v2, PL, BI, and PWM implementations reused directly",
        "- These are raw-interval spot checks, not 200-repetition coverage estimates",
        "- The full 17,600-group runner, atomic checkpoint, aggregation, plot renderer, and visual verifier remain available through the `all` stage",
        "",
        "| n | Quantile | Rep | Method | Lower | Upper | Width |",
        "| ---: | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lower = float(row["lower_bound"])
        upper = float(row["upper_bound"])
        report.append(
            f"| {row['sample_size']} | {float(row['critical_quantile']):.4f} | "
            f"{row['repetition']} | `{row['method']}` | {lower:.8g} | "
            f"{upper:.8g} | {upper - lower:.8g} |"
        )
    report_path = output_dir / "spot_check_report.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=("generate", "aggregate", "plot", "verify", "spot-check", "all"),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.stage == "spot-check":
        spot_check(args.output_dir, args.workers)
        return
    raw_path = args.output_dir / "raw_results.csv"
    if args.stage in {"generate", "all"}:
        raw_path = generate(args.output_dir, args.workers)
    if args.stage in {"aggregate", "all"}:
        write_aggregate(raw_path, args.output_dir)
    if args.stage in {"plot", "all"}:
        plot(raw_path, args.output_dir)
    if args.stage in {"verify", "all"}:
        verify(raw_path, args.output_dir)


if __name__ == "__main__":
    main()
