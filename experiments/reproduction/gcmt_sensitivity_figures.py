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
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
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


def _tasks(
    studies: tuple[str, ...],
    requested_methods: tuple[str, ...] = METHODS,
) -> list[Task]:
    tasks: list[Task] = []
    if "threshold" in studies:
        if "dro" in requested_methods:
            for region in REGIONS:
                tasks.extend(
                    Task("threshold", region, value, "dro")
                    for value in THRESHOLDS
                )
        for region in REGIONS:
            tasks.extend(
                Task("threshold", region, -1.0, method)
                for method in METHODS[1:]
                if method in requested_methods
            )
    if "exceedance" in studies:
        tasks.extend(
            Task("exceedance", region, quantile, method)
            for region in REGIONS
            for quantile in EXCEEDANCE_QUANTILES
            for method in METHODS
            if method in requested_methods
        )
    if "confidence" in studies:
        tasks.extend(
            Task("confidence", region, level, method)
            for region in REGIONS
            for level in CONFIDENCE_LEVELS
            for method in METHODS
            if method in requested_methods
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


def generate(
    output_dir: Path,
    studies: tuple[str, ...],
    workers: int,
    requested_methods: tuple[str, ...] = METHODS,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    requested_tasks = _tasks(studies, requested_methods)
    if not requested_tasks:
        raise ValueError("No tasks match the requested studies and methods")
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
    separate_methods_by_region = {
        "ECUADOR": ("pot_bt",),
        "OFF_COAST_OF_NORTHERN_CA": ("pot_bt",),
        "TURKEY": ("pot_bt",),
        "HOKKAIDO_JAPAN_REGION": ("pot_bt",),
        "BANDA_SEA": ("pot_bt",),
        "KURIL_ISLANDS": ("pot_bt",),
        "SOLOMON_ISLANDS": ("pot_bt",),
        "FIJI_ISLANDS_REGION": ("pot_bt",),
    }
    main_upper_by_row: dict[int, float] = {}
    mle_upper_by_row: dict[int, float] = {}
    for row in range(4):
        row_regions = REGIONS[2 * row : 2 * row + 2]
        main_upper = max(
            [
                lookup[("threshold", region, threshold, "dro")][1]
                for region in row_regions
                for threshold in THRESHOLDS
            ]
            + [
                lookup[("threshold", region, -1.0, method)][1]
                for region in row_regions
                for method in ("pl", "bayesian", "pwm")
            ]
        )
        mle_upper = max(
            lookup[("threshold", region, -1.0, "pot_bt")][1]
            for region in row_regions
        )
        main_upper_by_row[row] = 1.08 * main_upper
        mle_upper_by_row[row] = 1.08 * mle_upper

    # At 0.95\textwidth in the manuscript, this aspect ratio reproduces the
    # 207.24 pt vertical footprint of the Overleaf reference figure.
    figure = plt.figure(figsize=(10, 4.737))
    outer_grid = figure.add_gridspec(
        4,
        2,
        left=0.08,
        right=0.885,
        bottom=0.12,
        top=0.95,
        hspace=0.58,
        wspace=0.21,
    )
    for panel_index, region in enumerate(REGIONS):
        row, column = divmod(panel_index, 2)
        separate_methods = separate_methods_by_region.get(region, ())
        panel_grid = outer_grid[row, column].subgridspec(
            1,
            2,
            width_ratios=(7.0, 1.2),
            wspace=0.0,
        )
        axis = figure.add_subplot(panel_grid[0, 0])
        separate_axes: list[plt.Axes] = []
        if len(separate_methods) == 1:
            separate_axes = [figure.add_subplot(panel_grid[0, 1])]
        elif len(separate_methods) == 2:
            separate_grid = panel_grid[0, 1].subgridspec(
                1, 3, width_ratios=(0.6, 1.0, 1.0), wspace=0.0
            )
            separate_axes = [
                figure.add_subplot(separate_grid[0, index + 1])
                for index in range(len(separate_methods))
            ]
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
            fmt="none",
            capsize=2,
            linewidth=1.5,
            color="#1f77b4",
        )
        comparison_methods = tuple(
            method for method in METHODS[1:] if method not in separate_methods
        )
        benchmark_x = len(THRESHOLDS) + 2 + 3 * np.arange(
            len(comparison_methods)
        )
        for position, method in zip(benchmark_x, comparison_methods):
            center, error = _interval_center(
                lookup[("threshold", region, -1.0, method)]
            )
            _, color = LINE_STYLES[method]
            axis.errorbar(
                position,
                center,
                yerr=error,
                fmt="none",
                capsize=2,
                linewidth=1.5,
                color=color,
            )
        axis.axvline(len(THRESHOLDS) - 0.5, color="0.7", linestyle="--", linewidth=1)
        threshold_tick_indices = (0, 10, 20, 30)
        tick_positions = list(threshold_tick_indices)
        tick_labels = [f"{THRESHOLDS[index]:.2f}" for index in threshold_tick_indices]
        axis.set_xticks(tick_positions)
        # Keep the DRO threshold span and separator at the same physical
        # positions in every panel, regardless of how many EVT methods have
        # been moved to independent-scale mini-axes.
        axis.set_xlim(-2, 42)
        axis.set_ylim(0, main_upper_by_row[row])
        axis.tick_params(axis="y", labelsize=10)
        if panel_index >= 6:
            axis.set_xticklabels(tick_labels, rotation=0, ha="center")
            axis.tick_params(axis="x", labelsize=9, pad=1)
        else:
            axis.tick_params(axis="x", bottom=False, labelbottom=False)
        region_title = region.replace("_", " ")
        if region == "OFF_COAST_OF_NORTHERN_CA":
            region_title = "OFF COAST N. CALIFORNIA"
        axis.set_title(region_title, fontsize=13, pad=2)
        axis.grid(axis="y", alpha=0.25, linewidth=0.7)

        for separate_index, (separate_axis, method) in enumerate(
            zip(separate_axes, separate_methods)
        ):
            center, error = _interval_center(
                lookup[("threshold", region, -1.0, method)]
            )
            _, color = LINE_STYLES[method]
            separate_axis.errorbar(
                [0],
                [center],
                yerr=[error],
                fmt="none",
                capsize=2,
                linewidth=1.5,
                color=color,
            )
            separate_axis.set_xlim(-0.6, 0.6)
            separate_axis.set_ylim(0, mle_upper_by_row[row])
            separate_axis.set_xticks([])
            if len(separate_methods) == 2 and separate_index == 0:
                separate_axis.yaxis.tick_left()
            else:
                separate_axis.yaxis.tick_right()
            if row >= 1:
                scientific_formatter = ScalarFormatter(useMathText=True)
                scientific_formatter.set_powerlimits((-2, -2))
                separate_axis.yaxis.set_major_formatter(scientific_formatter)
                separate_axis.yaxis.set_offset_position("right")
                separate_axis.yaxis.get_offset_text().set_fontsize(6.5)
            separate_axis.tick_params(axis="y", labelsize=7.5, pad=1)
            separate_axis.grid(axis="y", alpha=0.25, linewidth=0.7)
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=LINE_STYLES[method][1],
            linewidth=2,
            label=METHOD_LABELS[method],
        )
        for method in ("pl", "bayesian", "pwm", "pot_bt")
    ]
    figure.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(0.905, 0.53),
        ncol=1,
        frameon=False,
        fontsize=8.5,
        handlelength=1.4,
        labelspacing=0.8,
    )
    figure.supylabel(
        r"Exceedance probability $P(X \geq M_W)$",
        fontsize=13,
        x=0.012,
    )
    path = output_dir / "real_data_vs_evt_threshold_percentage.png"
    figure.savefig(path, dpi=200)
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
    stacked_layout = study in {"exceedance", "confidence"}
    if stacked_layout:
        figure_height = 6.5 if study == "exceedance" else 10.0
        figure, axes = plt.subplots(
            4, 4, figsize=(12, figure_height), sharex=False
        )
        panel_pairs = [
            (axes[2 * (index // 4), index % 4], axes[2 * (index // 4) + 1, index % 4])
            for index in range(len(REGIONS))
        ]
    else:
        figure, axes = plt.subplots(2, 8, figsize=(30, 5), sharex="col")
        panel_pairs = [(axes[0, index], axes[1, index]) for index in range(len(REGIONS))]

    mle_axis_pairs: list[tuple[plt.Axes, plt.Axes]] = []
    for region_index, (region, (bound_axis, width_axis)) in enumerate(
        zip(REGIONS, panel_pairs)
    ):
        if study in {"exceedance", "confidence"}:
            mle_bound_axis = bound_axis.twinx()
            mle_width_axis = width_axis.twinx()
            mle_axis_pairs.append((mle_bound_axis, mle_width_axis))
        else:
            mle_bound_axis = bound_axis
            mle_width_axis = width_axis
        for method in METHODS:
            bounds = [lookup[(study, region, setting, method)] for setting in settings]
            lower = np.asarray([pair[0] for pair in bounds])
            upper = np.asarray([pair[1] for pair in bounds])
            style, color = LINE_STYLES[method]
            method_bound_axis = mle_bound_axis if method == "pot_bt" else bound_axis
            method_width_axis = mle_width_axis if method == "pot_bt" else width_axis
            method_bound_axis.plot(
                settings,
                upper,
                linestyle=style,
                color=color,
                linewidth=2 if method == "dro" else 1.2,
                label=METHOD_LABELS[method],
            )
            method_bound_axis.scatter(
                settings, lower, color=color, s=7, alpha=0.55
            )
            method_width_axis.plot(
                settings,
                upper - lower,
                linestyle=style,
                color=color,
                linewidth=2 if method == "dro" else 1.2,
            )
        region_title = region.replace("_", " ")
        if region == "OFF_COAST_OF_NORTHERN_CA":
            region_title = "OFF COAST N. CALIFORNIA"
        bound_axis.set_title(region_title, fontsize=15, pad=2)
        bound_axis.grid(alpha=0.2)
        width_axis.grid(alpha=0.2)
        bound_axis.set_ylim(bottom=0)
        width_axis.set_ylim(bottom=0)
        if study in {"exceedance", "confidence"}:
            _, mle_color = LINE_STYLES["pot_bt"]
            for mle_axis in (mle_bound_axis, mle_width_axis):
                mle_axis.set_ylim(bottom=0)
                mle_axis.tick_params(
                    axis="y",
                    colors=mle_color,
                    labelsize=11 if study == "confidence" else 8,
                    pad=1,
                )
                mle_axis.spines["right"].set_color(mle_color)

        if stacked_layout:
            tick_fontsize = 13 if study == "confidence" else 11
            label_fontsize = 12 if study == "confidence" else 10
            bound_axis.tick_params(axis="both", labelsize=tick_fontsize)
            bound_axis.tick_params(axis="x", bottom=False, labelbottom=False)
            width_axis.tick_params(axis="both", labelsize=tick_fontsize)
            if study == "exceedance":
                x_ticks = (0.990, 0.995, 0.9995)
                x_labels = [f"{value:.4g}" for value in x_ticks]
            else:
                x_ticks = (0.90, 0.95, 0.99)
                x_labels = [f"{value:.2f}" for value in x_ticks]
            width_axis.set_xticks(x_ticks, x_labels)
            if region_index % 4 == 0:
                bound_axis.set_ylabel(
                    "Confidence bound", fontsize=label_fontsize, labelpad=2
                )
                width_axis.set_ylabel(
                    "Interval width", fontsize=label_fontsize, labelpad=2
                )

    if not stacked_layout:
        axes[0, 0].set_ylabel("Confidence bound")
        axes[1, 0].set_ylabel("Confidence interval width")

    handles, labels = panel_pairs[0][0].get_legend_handles_labels()
    if study in {"exceedance", "confidence"}:
        mle_handles, mle_labels = mle_axis_pairs[0][0].get_legend_handles_labels()
        handles = [handles[0], *mle_handles, *handles[1:]]
        labels = [labels[0], *mle_labels, *labels[1:]]
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        frameon=True,
        fontsize=(13 if study == "confidence" else 11) if stacked_layout else None,
    )
    if stacked_layout:
        figure.tight_layout(rect=(0, 0.075, 1, 1), pad=0.6, h_pad=0.35, w_pad=0.8)
        for row in (1, 3):
            for column, axis in enumerate(axes[row, :]):
                position = axis.get_position()
                axis.set_position(
                    [
                        position.x0,
                        position.y0 + 0.015,
                        position.width,
                        position.height,
                    ]
                )
                if study in {"exceedance", "confidence"}:
                    mle_axis_pairs[(row // 2) * 4 + column][1].set_position(
                        axis.get_position()
                    )
    else:
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


def _parse_methods(value: str) -> tuple[str, ...]:
    if value == "all":
        return METHODS
    requested = tuple(part.strip() for part in value.split(",") if part.strip())
    invalid = set(requested) - set(METHODS)
    if not requested or invalid:
        raise ValueError(f"Invalid methods: {sorted(invalid)}")
    return requested


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("generate", "plot", "verify", "spot-check-bi", "all")
    )
    parser.add_argument("--studies", default="all")
    parser.add_argument(
        "--methods",
        default="all",
        help="Comma-separated internal method names, e.g. pot_bt,pwm",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.stage == "spot-check-bi":
        spot_check_bayesian(args.output_dir, args.workers)
        return
    studies = _parse_studies(args.studies)
    methods = _parse_methods(args.methods)
    if methods != METHODS and args.stage != "generate":
        raise ValueError(
            "Method-filtered runs support only the generate stage; merge the "
            "result with unchanged method rows before plotting."
        )
    raw_path = args.output_dir / "raw_results.csv"
    if args.stage in {"generate", "all"}:
        raw_path = generate(args.output_dir, studies, args.workers, methods)
    if args.stage in {"plot", "all"}:
        plot(raw_path, args.output_dir, studies)
    if args.stage == "verify":
        verify(args.output_dir, studies)


if __name__ == "__main__":
    main()
