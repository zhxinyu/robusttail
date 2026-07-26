"""Reproduce GCMT sensitivity Figures 3, 4, and 9 end to end.

The numerical calls are factored from ``experiments/run_scripts/exp_cmt.py``
and continue to use the established DRO and EVT implementations.
"""

from __future__ import annotations

import argparse
import csv
import logging
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
DEFAULT_OUTPUT_DIR = (
    REPOSITORY_ROOT / "experiments" / "generated" / "gcmt_sensitivity_figures"
)
MANUSCRIPT_PLOT_DIR = REPOSITORY_ROOT.parent / "latex" / "plots"

RANDOM_SEED = 20220222
BOOTSTRAPPING_SIZE = 500
ELLIPSOIDAL_DIMENSION = 3
DEFAULT_WORKERS = min(16, os.cpu_count() or 1)
MAX_TASKS_PER_WORKER = 10

REGIONS = (
    "ECUADOR",
    "OFF_COAST_OF_NORTHERN_CA",
    "TURKEY",
    "HOKKAIDO_JAPAN_REGION",
    "BANDA_SEA",
    "KURIL_ISLANDS",
    "SOLOMON_ISLANDS",
    "FIJI_ISLANDS_REGION",
)
METHODS = ("dro", "pot_bt", "pl", "bayesian", "pwm")
METHOD_LABELS = {
    "dro": "DRO",
    "pot_bt": "MLE-v2",
    "pl": "PL",
    "bayesian": "BI",
    "pwm": "PWM",
}
THRESHOLDS = tuple(np.linspace(0.60, 0.90, 31))
EXCEEDANCE_QUANTILES = (
    0.990,
    0.991,
    0.992,
    0.993,
    0.994,
    0.995,
    0.996,
    0.997,
    0.998,
    0.999,
    0.9991,
    0.9992,
    0.9993,
    0.9995,
)
CONFIDENCE_LEVELS = tuple(np.linspace(0.90, 0.99, 10))
FIELDS = (
    "study",
    "region",
    "setting",
    "method",
    "lower_bound",
    "upper_bound",
)

_STANDARDIZED_DATA: dict[str, np.ndarray] = {}


@dataclass(frozen=True)
class Task:
    study: str
    region: str
    setting: float
    method: str

    @property
    def key(self) -> tuple[str, str, float, str]:
        return self.study, self.region, self.setting, self.method


BI_SPOT_TASKS = (
    Task("threshold", "ECUADOR", -1.0, "bayesian"),
    Task("exceedance", "ECUADOR", 0.9993, "bayesian"),
    Task("confidence", "ECUADOR", 0.99, "bayesian"),
)


def _load_standardized_data() -> dict[str, np.ndarray]:
    frame = parse_ndk()
    return {
        region: np.asarray(
            zscore(frame.loc[frame["location"] == region, "Mw"].to_numpy(dtype=float)),
            dtype=float,
        )
        for region in REGIONS
    }


def _worker_initialize() -> None:
    global _STANDARDIZED_DATA
    _STANDARDIZED_DATA = _load_standardized_data()


def _tasks(studies: tuple[str, ...]) -> list[Task]:
    tasks: list[Task] = []
    if "threshold" in studies:
        for region in REGIONS:
            tasks.extend(Task("threshold", region, value, "dro") for value in THRESHOLDS)
            tasks.extend(Task("threshold", region, -1.0, method) for method in METHODS[1:])
    if "exceedance" in studies:
        tasks.extend(
            Task("exceedance", region, quantile, method)
            for region in REGIONS
            for quantile in EXCEEDANCE_QUANTILES
            for method in METHODS
        )
    if "confidence" in studies:
        tasks.extend(
            Task("confidence", region, level, method)
            for region in REGIONS
            for level in CONFIDENCE_LEVELS
            for method in METHODS
        )
    return tasks


def _run_task(task: Task) -> dict[str, object]:
    values = _STANDARDIZED_DATA[task.region]
    if task.study == "threshold":
        target_quantile = 0.999
        threshold = task.setting if task.method == "dro" else 0.70
        alpha = 0.05
    elif task.study == "exceedance":
        target_quantile = task.setting
        threshold = 0.70
        alpha = 0.05
    elif task.study == "confidence":
        target_quantile = 0.999
        threshold = 0.70
        alpha = 1.0 - task.setting
    else:
        raise ValueError(task.study)
    objective = float(np.quantile(values, target_quantile))

    if task.method == "dro":
        lower, upper = estimate_tail_probability_D2_chi2_only(
            input_data=values,
            left_end_point_objective=objective,
            right_end_point_objective=np.inf,
            threshold_percentage=threshold,
            g_ellipsoidal_dimension=ELLIPSOIDAL_DIMENSION,
            alpha=alpha,
            random_state=RANDOM_SEED,
            bootstrapping_size=BOOTSTRAPPING_SIZE,
            right_endpoint=np.inf,
        )
    else:
        lower, upper = benchmark_estimate_tail_probability(
            input_data=values,
            left_end_point_objective=objective,
            right_end_point_objective=np.inf,
            method=task.method,
            alpha=alpha,
            random_state=RANDOM_SEED,
        )
    return {
        "study": task.study,
        "region": task.region,
        "setting": task.setting,
        "method": task.method,
        "lower_bound": float(lower),
        "upper_bound": float(upper),
    }


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
                str(row["study"]),
                str(row["region"]),
                float(row["setting"]),
                str(row["method"]),
            )
        ],
    )
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(ordered)
    temporary.replace(path)


def generate(output_dir: Path, studies: tuple[str, ...], workers: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    requested_tasks = _tasks(studies)
    order = {task.key: index for index, task in enumerate(requested_tasks)}
    final_path = output_dir / "raw_results.csv"
    partial_path = output_dir / "raw_results.partial.csv"
    existing = _read(final_path) or _read(partial_path)
    rows: list[dict[str, object]] = [dict(row) for row in existing]
    completed = {
        (row["study"], row["region"], float(row["setting"]), row["method"])
        for row in existing
    }
    if not completed.issubset(order):
        raise ValueError("Output directory contains results outside requested studies")
    pending = [task for task in requested_tasks if task.key not in completed]
    LOGGER.info("%d/%d tasks complete; %d pending", len(completed), len(order), len(pending))
    started = time.monotonic()

    context = multiprocessing.get_context("spawn")
    with context.Pool(
        processes=workers,
        maxtasksperchild=MAX_TASKS_PER_WORKER,
        initializer=_worker_initialize,
    ) as pool:
        for completed_now, result in enumerate(
            pool.imap_unordered(_run_task, pending, chunksize=1), start=1
        ):
            rows.append(result)
            done = len(existing) + completed_now
            if completed_now == 1 or done % max(1, len(order) // 100) == 0 or done == len(order):
                _write_atomic(partial_path, rows, order)
                elapsed = time.monotonic() - started
                rate = completed_now / max(elapsed, 1e-9)
                LOGGER.info(
                    "Progress %d/%d (%.1f%%); elapsed %.1fs; ETA %.1fs",
                    done,
                    len(order),
                    100 * done / len(order),
                    elapsed,
                    (len(order) - done) / rate,
                )

    _write_atomic(final_path, rows, order)
    partial_path.unlink(missing_ok=True)
    return final_path


def _lookup(raw_path: Path) -> dict[tuple[str, str, float, str], tuple[float, float]]:
    return {
        (
            row["study"],
            row["region"],
            float(row["setting"]),
            row["method"],
        ): (float(row["lower_bound"]), float(row["upper_bound"]))
        for row in _read(raw_path)
    }


def _interval_center(bounds: tuple[float, float]) -> tuple[float, float]:
    lower, upper = bounds
    return (lower + upper) / 2, (upper - lower) / 2


def plot_threshold(raw_path: Path, output_dir: Path) -> Path:
    lookup = _lookup(raw_path)
    figure, axes = plt.subplots(4, 2, figsize=(12, 9), sharex=False)
    for axis, region in zip(axes.flat, REGIONS):
        centers, errors = zip(
            *[
                _interval_center(lookup[("threshold", region, threshold, "dro")])
                for threshold in THRESHOLDS
            ]
        )
        axis.errorbar(
            np.arange(len(THRESHOLDS)),
            centers,
            yerr=errors,
            fmt="o",
            markersize=3.5,
            linewidth=1.2,
            color="#1f77b4",
        )
        benchmark_x = np.arange(len(THRESHOLDS) + 1, len(THRESHOLDS) + 5)
        benchmark = [
            _interval_center(lookup[("threshold", region, -1.0, method)])
            for method in METHODS[1:]
        ]
        axis.errorbar(
            benchmark_x,
            [value[0] for value in benchmark],
            yerr=[value[1] for value in benchmark],
            fmt="s",
            markersize=4,
            linewidth=1.2,
            color="black",
        )
        axis.axvline(len(THRESHOLDS) - 0.5, color="0.7", linestyle="--", linewidth=1)
        tick_positions = list(range(0, len(THRESHOLDS), 5)) + list(benchmark_x)
        tick_labels = [f"{THRESHOLDS[index]:.1f}" for index in range(0, len(THRESHOLDS), 5)]
        tick_labels += [METHOD_LABELS[method] for method in METHODS[1:]]
        axis.set_xticks(tick_positions, tick_labels, rotation=45, ha="right")
        axis.set_title(region.replace("_", " "))
        axis.grid(axis="y", alpha=0.25)
    figure.supylabel(r"Exceedance probability $P(X \geq M_W)$")
    figure.tight_layout()
    path = output_dir / "real_data_vs_evt_threshold_percentage.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


LINE_STYLES = {
    "dro": ("-", "#1f77b4"),
    "pot_bt": ("--", "#ff7f0e"),
    "pl": ("-.", "#2ca02c"),
    "bayesian": (":", "#ff6666"),
    "pwm": ((0, (3, 1, 1, 1)), "#9467bd"),
}


def _plot_grid(
    raw_path: Path,
    output_dir: Path,
    study: str,
    settings: tuple[float, ...],
    filename: str,
) -> Path:
    lookup = _lookup(raw_path)
    figure, axes = plt.subplots(2, 8, figsize=(30, 5), sharex="col")
    for column, region in enumerate(REGIONS):
        for method in METHODS:
            bounds = [lookup[(study, region, setting, method)] for setting in settings]
            lower = np.asarray([pair[0] for pair in bounds])
            upper = np.asarray([pair[1] for pair in bounds])
            style, color = LINE_STYLES[method]
            axes[0, column].plot(
                settings,
                upper,
                linestyle=style,
                color=color,
                linewidth=2 if method == "dro" else 1.2,
                label=METHOD_LABELS[method],
            )
            axes[0, column].scatter(settings, lower, color=color, s=7, alpha=0.55)
            axes[1, column].plot(
                settings,
                upper - lower,
                linestyle=style,
                color=color,
                linewidth=2 if method == "dro" else 1.2,
            )
        axes[0, column].set_title(region.replace("_", " "))
        axes[0, column].grid(alpha=0.2)
        axes[1, column].grid(alpha=0.2)
    axes[0, 0].set_ylabel("Confidence bound")
    axes[1, 0].set_ylabel("Confidence interval width")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=5, frameon=True)
    figure.tight_layout(rect=(0, 0.08, 1, 1))
    path = output_dir / filename
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot(raw_path: Path, output_dir: Path, studies: tuple[str, ...]) -> list[Path]:
    paths: list[Path] = []
    if "threshold" in studies:
        paths.append(plot_threshold(raw_path, output_dir))
    if "exceedance" in studies:
        paths.append(
            _plot_grid(
                raw_path,
                output_dir,
                "exceedance",
                EXCEEDANCE_QUANTILES,
                "real_data_vs_evt_critical_values.png",
            )
        )
    if "confidence" in studies:
        paths.append(
            _plot_grid(
                raw_path,
                output_dir,
                "confidence",
                CONFIDENCE_LEVELS,
                "real_data_vs_evt_confidence_levels.png",
            )
        )
    return paths


def verify(output_dir: Path, studies: tuple[str, ...]) -> Path:
    mappings = {
        "threshold": "real_data_vs_evt_threshold_percentage.png",
        "exceedance": "real_data_vs_evt_critical_values.png",
        "confidence": "real_data_vs_evt_confidence_levels.png",
    }
    raw_path = output_dir / "raw_results.csv"
    rows = _read(raw_path)
    expected_tasks = _tasks(studies)
    expected_keys = {task.key for task in expected_tasks}
    actual_keys = {
        (row["study"], row["region"], float(row["setting"]), row["method"])
        for row in rows
    }
    if len(rows) != len(expected_tasks) or actual_keys != expected_keys:
        raise ValueError(
            f"Raw result grid is incomplete: {len(rows)}/{len(expected_tasks)} rows"
        )
    bounds = np.asarray(
        [
            (float(row["lower_bound"]), float(row["upper_bound"]))
            for row in rows
        ],
        dtype=float,
    )
    if not np.isfinite(bounds).all():
        bad = int((~np.isfinite(bounds)).any(axis=1).sum())
        raise ValueError(f"{bad} result rows contain non-finite interval bounds")
    if np.any(bounds[:, 0] > bounds[:, 1]):
        bad = int((bounds[:, 0] > bounds[:, 1]).sum())
        raise ValueError(f"{bad} result rows have lower bound above upper bound")

    report = [
        "# GCMT sensitivity-figure verification report",
        "",
        "- Numerical implementations: existing DRO and EVT functions factored from `exp_cmt.py`",
        "- Inputs: existing raw NDK parser followed by the script's regional z-score transformation",
        "- DRO bootstrap calibration: 500; seed: 20220222; NumPy-era bootstrap sequence",
        "- R-based conventional estimators: explicit seed 20220222, including deterministic BI MCMC",
        f"- Complete numerical grid: {len(rows)}/{len(expected_tasks)} finite, ordered intervals",
        "- Plot criterion: substantive scientific and visual equivalence; PNG byte equality is not required",
        "",
        "| Study | Display | Numerical rows | Generated pixels | Manuscript pixels |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    display = {"threshold": "Figure 3", "exceedance": "Figure 4", "confidence": "Figure 9"}
    for study in studies:
        filename = mappings[study]
        generated = output_dir / filename
        expected = MANUSCRIPT_PLOT_DIR / filename
        if not generated.exists() or not expected.exists():
            raise FileNotFoundError(filename)
        generated_shape = plt.imread(generated).shape
        expected_shape = plt.imread(expected).shape
        study_rows = sum(row["study"] == study for row in rows)
        report.append(
            f"| `{study}` | {display[study]} | {study_rows} | "
            f"{generated_shape[1]}×{generated_shape[0]} | "
            f"{expected_shape[1]}×{expected_shape[0]} |"
        )
    report.extend(
        [
            "",
            "Acceptance requires matching regions, parameter grids, methods, interval bounds, trends, and panel structure. Visual review is still required before a study is marked accepted.",
        ]
    )
    path = output_dir / "verification_report.md"
    path.write_text("\n".join(report) + "\n", encoding="utf-8")
    return path


def spot_check_bayesian(output_dir: Path, workers: int) -> Path:
    """Replay one seeded BI interval from each sensitivity figure."""
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = list(BI_SPOT_TASKS)
    repeated = tasks + tasks
    context = multiprocessing.get_context("spawn")
    with context.Pool(
        processes=min(workers, len(repeated)),
        maxtasksperchild=1,
        initializer=_worker_initialize,
    ) as pool:
        results = list(pool.imap(_run_task, repeated, chunksize=1))
    first, replay = results[: len(tasks)], results[len(tasks) :]
    differences = [
        max(
            abs(float(original[field]) - float(repeated_row[field]))
            for field in ("lower_bound", "upper_bound")
        )
        for original, repeated_row in zip(first, replay)
    ]
    path = output_dir / "bayesian_spot_check_results.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=(*FIELDS, "replay_max_difference"))
        writer.writeheader()
        for row, difference in zip(first, differences):
            writer.writerow({**row, "replay_max_difference": difference})
    finite = sum(
        np.isfinite([float(row["lower_bound"]), float(row["upper_bound"])]).all()
        for row in first
    )
    ordered = sum(
        float(row["lower_bound"]) <= float(row["upper_bound"]) for row in first
    )
    exact = sum(difference <= 1e-12 for difference in differences)
    report = [
        "# Figures 3, 4, and 9 seeded-BI spot check",
        "",
        f"- Representative Bayesian intervals: {len(first)}",
        f"- Finite intervals: {finite}/{len(first)}",
        f"- Ordered intervals: {ordered}/{len(first)}",
        f"- Independent seeded replays within `1e-12`: {exact}/{len(first)}",
        "- One active setting is checked for each threshold, exceedance-level, and confidence-level figure",
        "- This focused replay closes the deterministic BI audit; the accepted 1,240-interval visual run is documented separately",
        "",
        "| Study | Region | Setting | Lower | Upper | Replay difference |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row, difference in zip(first, differences):
        report.append(
            f"| `{row['study']}` | `{row['region']}` | {float(row['setting']):.6g} | "
            f"{float(row['lower_bound']):.10g} | {float(row['upper_bound']):.10g} | "
            f"{difference:.3g} |"
        )
    report_path = output_dir / "bayesian_spot_check_report.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    return report_path


def _parse_studies(value: str) -> tuple[str, ...]:
    if value == "all":
        return ("threshold", "exceedance", "confidence")
    requested = tuple(part.strip() for part in value.split(",") if part.strip())
    invalid = set(requested) - {"threshold", "exceedance", "confidence"}
    if not requested or invalid:
        raise ValueError(f"Invalid studies: {sorted(invalid)}")
    return requested


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("generate", "plot", "verify", "spot-check-bi", "all")
    )
    parser.add_argument("--studies", default="all")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.stage == "spot-check-bi":
        spot_check_bayesian(args.output_dir, args.workers)
        return
    studies = _parse_studies(args.studies)
    raw_path = args.output_dir / "raw_results.csv"
    if args.stage in {"generate", "all"}:
        raw_path = generate(args.output_dir, studies, args.workers)
    if args.stage in {"plot", "all"}:
        plot(raw_path, args.output_dir, studies)
    if args.stage in {"verify", "all"}:
        verify(args.output_dir, studies)


if __name__ == "__main__":
    main()
