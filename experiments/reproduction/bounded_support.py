"""Reproduce bounded-support Figure 2 and Table I.1 end to end.

This is a checkpointed, manuscript-facing factorization of
``experiments/run_scripts/exp_bound_support.py``.  It preserves that script's
seed, sampling order (including the two non-displayed endpoint settings that
advance the random stream), solver, and 200 repetitions per displayed cell.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import multiprocessing
import os
import re
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
from scipy.stats import genpareto

from experiments.run_scripts.tail_probability.tail_probability_estimation import (
    estimate_tail_probability_D2_chi2_only,
)

LOGGER = logging.getLogger(__name__)
logging.getLogger("rpy2.rinterface_lib.callbacks").setLevel(logging.ERROR)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MANUSCRIPT_PATH = REPOSITORY_ROOT.parent / "latex" / "jasa_manu.tex"
MANUSCRIPT_FIGURE = REPOSITORY_ROOT.parent / "latex" / "plots" / "bounded_support.png"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "experiments" / "generated" / "bounded_support"

RANDOM_SEED = 20220222
REPETITIONS = 200
SAMPLE_SIZE = 500
SHAPE = -0.5
THRESHOLD_PERCENTAGE = 0.70
BOOTSTRAPPING_SIZE = 500
ELLIPSOIDAL_DIMENSION = 3
ALPHA = 0.05
QUANTILES = (0.95, 0.99, 0.995)
# The original script generated all twelve groups in this order.  The 1.90 and
# 2.01 groups were not retained in the manuscript, but their draws must still
# advance NumPy's seeded stream to reproduce every later displayed group.
GENERATION_ENDPOINTS = (1.90, 1.91, 1.93, 1.95, 1.98, 2.00, 2.01, 2.03, 2.05, 2.08, 2.10, math.inf)
DISPLAY_ENDPOINTS = (1.91, 1.93, 1.95, 1.98, 2.00, 2.03, 2.05, 2.08, 2.10, math.inf)
DEFAULT_WORKERS = min(16, os.cpu_count() or 1)
MAX_TASKS_PER_WORKER = 25
FIELDS = ("quantile", "endpoint", "repetition", "lower_bound", "upper_bound")


@dataclass(frozen=True)
class Task:
    quantile: float
    endpoint: float
    repetition: int
    sample: np.ndarray

    @property
    def key(self) -> tuple[float, float, int]:
        return self.quantile, self.endpoint, self.repetition


def _all_tasks() -> list[Task]:
    """Regenerate the original legacy NumPy data stream exactly."""
    np.random.seed(RANDOM_SEED)
    tasks: list[Task] = []
    displayed = set(DISPLAY_ENDPOINTS)
    for quantile in QUANTILES:
        for endpoint in GENERATION_ENDPOINTS:
            samples = [
                np.asarray(genpareto.rvs(size=SAMPLE_SIZE, c=SHAPE, loc=0, scale=1))
                for _ in range(REPETITIONS)
            ]
            if endpoint in displayed:
                tasks.extend(
                    Task(quantile, endpoint, repetition, sample)
                    for repetition, sample in enumerate(samples)
                )
    return tasks


def _run_task(task: Task) -> dict[str, object]:
    objective = float(genpareto.ppf(q=task.quantile, c=SHAPE, loc=0, scale=1))
    lower, upper = estimate_tail_probability_D2_chi2_only(
        input_data=task.sample,
        left_end_point_objective=objective,
        right_end_point_objective=np.inf,
        threshold_percentage=THRESHOLD_PERCENTAGE,
        g_ellipsoidal_dimension=ELLIPSOIDAL_DIMENSION,
        alpha=ALPHA,
        random_state=RANDOM_SEED,
        bootstrapping_size=BOOTSTRAPPING_SIZE,
        right_endpoint=task.endpoint,
    )
    return {
        "quantile": task.quantile,
        "endpoint": task.endpoint,
        "repetition": task.repetition,
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
            (float(row["quantile"]), float(row["endpoint"]), int(row["repetition"]))
        ],
    )
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(ordered)
    temporary.replace(path)


def generate(output_dir: Path, workers: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = _all_tasks()
    order = {task.key: index for index, task in enumerate(tasks)}
    final_path = output_dir / "raw_results.csv"
    partial_path = output_dir / "raw_results.partial.csv"
    existing = _read(final_path) or _read(partial_path)
    rows: list[dict[str, object]] = [dict(row) for row in existing]
    completed = {
        (float(row["quantile"]), float(row["endpoint"]), int(row["repetition"]))
        for row in existing
    }
    if not completed.issubset(order):
        raise ValueError("Output directory contains results outside the manuscript grid")
    pending = [task for task in tasks if task.key not in completed]
    LOGGER.info(
        "%d/%d interval estimates complete; %d pending",
        len(completed),
        len(tasks),
        len(pending),
    )
    started = time.monotonic()
    context = multiprocessing.get_context("spawn")
    with context.Pool(
        processes=workers,
        maxtasksperchild=MAX_TASKS_PER_WORKER,
    ) as pool:
        for completed_now, result in enumerate(
            pool.imap_unordered(_run_task, pending, chunksize=1), start=1
        ):
            rows.append(result)
            done = len(existing) + completed_now
            if completed_now == 1 or done % max(1, len(tasks) // 100) == 0 or done == len(tasks):
                _write_atomic(partial_path, rows, order)
                elapsed = time.monotonic() - started
                rate = completed_now / max(elapsed, 1e-9)
                LOGGER.info(
                    "Progress %d/%d (%.1f%%); elapsed %.1fs; ETA %.1fs",
                    done,
                    len(tasks),
                    100 * done / len(tasks),
                    elapsed,
                    (len(tasks) - done) / rate,
                )
    _write_atomic(final_path, rows, order)
    partial_path.unlink(missing_ok=True)
    return final_path


def _half_width(values: np.ndarray, *, ddof: int) -> float:
    """Return the original script's normal Monte Carlo half-width.

    The retained analysis mixed pandas' sample standard deviation for the
    bounds with NumPy's population standard deviation for coverage and width.
    Preserve that historical convention because these are manuscript-facing
    reproduction values.
    """
    return float(1.96 * np.std(values, ddof=ddof) / math.sqrt(values.size))


def aggregate(raw_path: Path) -> list[dict[str, float]]:
    grouped: dict[tuple[float, float], list[tuple[float, float]]] = {}
    for row in _read(raw_path):
        key = float(row["quantile"]), float(row["endpoint"])
        grouped.setdefault(key, []).append(
            (float(row["lower_bound"]), float(row["upper_bound"]))
        )
    expected = {(q, endpoint) for q in QUANTILES for endpoint in DISPLAY_ENDPOINTS}
    if set(grouped) != expected or any(len(values) != REPETITIONS for values in grouped.values()):
        raise ValueError("Raw results are incomplete")
    results: list[dict[str, float]] = []
    for quantile in QUANTILES:
        true_value = 1.0 - quantile
        for endpoint in DISPLAY_ENDPOINTS:
            bounds = np.asarray(grouped[(quantile, endpoint)], dtype=float)
            if (
                bounds.shape != (REPETITIONS, 2)
                or not np.isfinite(bounds).all()
                or np.any(bounds[:, 0] > bounds[:, 1])
            ):
                raise ValueError(
                    f"Invalid interval results for {(quantile, endpoint)}"
                )
            lower, upper = bounds[:, 0], bounds[:, 1]
            coverage = ((lower <= true_value) & (true_value <= upper)).astype(float)
            width = upper - lower
            results.append(
                {
                    "quantile": quantile,
                    "endpoint": endpoint,
                    "coverage": float(np.mean(coverage)),
                    "coverage_half_width": _half_width(coverage, ddof=0),
                    "lower": float(np.mean(lower)),
                    "lower_half_width": _half_width(lower, ddof=1),
                    "upper": float(np.mean(upper)),
                    "upper_half_width": _half_width(upper, ddof=1),
                    "width": float(np.mean(width)),
                    "width_half_width": _half_width(width, ddof=0),
                }
            )
    return results


def _scientific(value: float) -> str:
    exponent = math.floor(math.log10(abs(value)))
    coefficient = value / (10**exponent)
    if round(abs(coefficient), 2) >= 10:
        coefficient /= 10
        exponent += 1
    return rf"{coefficient:.2f}\times 10^{{{exponent}}}"


def _endpoint_tex(endpoint: float) -> str:
    return r"\infty" if math.isinf(endpoint) else f"{endpoint:.2f}"


def render(raw_path: Path, output_dir: Path) -> Path:
    rows = aggregate(raw_path)
    lines: list[str] = []
    for index, row in enumerate(rows):
        quantile = row["quantile"]
        prefix = (
            rf"\multirow{{10}}{{*}}{{${{P(X\geq q_{{{quantile:g}}})}}$}} "
            if index % len(DISPLAY_ENDPOINTS) == 0
            else ""
        )
        line = (
            f"{prefix}& ${_endpoint_tex(row['endpoint'])}$"
            rf" & ${row['coverage']:.2f} (\pm {row['coverage_half_width']:.2f})$"
            rf" & ${_scientific(row['lower'])}(\pm {_scientific(row['lower_half_width'])})$"
            rf" & ${_scientific(row['upper'])}(\pm {_scientific(row['upper_half_width'])})$"
            rf" & ${_scientific(row['width'])}(\pm {_scientific(row['width_half_width'])})$ \\"
        )
        if (index + 1) % len(DISPLAY_ENDPOINTS) == 0:
            line += r"\hline"
        lines.append(line)
    path = output_dir / "table_i_1_rows.tex"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def plot(raw_path: Path, output_dir: Path) -> Path:
    rows = aggregate(raw_path)
    lookup = {(row["quantile"], row["endpoint"]): row for row in rows}
    infinity_replacement = 2.12
    x = np.asarray(
        [
            infinity_replacement if math.isinf(value) else value
            for value in DISPLAY_ENDPOINTS
        ],
        dtype=float,
    )
    labels = [
        "\N{INFINITY}" if math.isinf(value) else str(value)
        for value in DISPLAY_ENDPOINTS
    ]
    figure, axes = plt.subplots(
        3,
        3,
        figsize=(18, 9),
        sharex=True,
        sharey="row",
    )
    for column, quantile in enumerate(QUANTILES):
        selected = [lookup[(quantile, endpoint)] for endpoint in DISPLAY_ENDPOINTS]
        true_value = 1.0 - quantile
        coverage = axes[0, column].errorbar(
            x,
            [row["coverage"] for row in selected],
            yerr=[row["coverage_half_width"] for row in selected],
            color="tab:blue",
            marker="o",
            markersize=5,
            linewidth=2,
            elinewidth=1.5,
            capsize=4,
            label="Empirical coverage",
        )
        nominal = axes[0, column].axhline(
            0.95,
            color="tab:red",
            linestyle=":",
            alpha=0.85,
            linewidth=2,
            label="Nominal coverage 0.95",
        )
        title = rf"$P(X\geq q_{{{quantile:g}}})$"
        axes[0, column].set_title(title, fontsize=20, pad=7)
        lower = axes[1, column].errorbar(
            x,
            [row["lower"] for row in selected],
            yerr=[row["lower_half_width"] for row in selected],
            color="tab:orange",
            marker="o",
            markersize=5,
            linewidth=2,
            elinewidth=1.5,
            capsize=4,
            label="Lower bound",
        )
        upper = axes[1, column].errorbar(
            x,
            [row["upper"] for row in selected],
            yerr=[row["upper_half_width"] for row in selected],
            color="tab:green",
            marker="^",
            markersize=6,
            linewidth=2,
            elinewidth=1.5,
            capsize=4,
            label="Upper bound",
        )
        truth = axes[1, column].axhline(
            true_value,
            color="tab:red",
            linestyle=":",
            alpha=0.85,
            linewidth=2,
            label="True probability",
        )
        width = axes[2, column].errorbar(
            x,
            [row["width"] for row in selected],
            yerr=[row["width_half_width"] for row in selected],
            color="tab:purple",
            marker="o",
            markersize=5,
            linewidth=2,
            elinewidth=1.5,
            capsize=4,
            label="Interval width",
        )
        for row_index in range(3):
            axes[row_index, column].grid(
                axis="both",
                color="0.75",
                alpha=0.35,
                linewidth=0.8,
            )
            axes[row_index, column].set_xticks(x, labels)
            axes[row_index, column].tick_params(
                axis="both",
                labelsize=14,
                pad=3,
                labelbottom=row_index == 2,
            )
            if row_index == 2:
                for tick in axes[row_index, column].get_xticklabels():
                    tick.set_rotation(35)
                    tick.set_horizontalalignment("right")
                    if tick.get_text() == "2.0":
                        tick.set_weight("bold")
        axes[0, column].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        probability_ticks = np.arange(0.0, 0.071, 0.01)
        axes[1, column].set_yticks(probability_ticks)
        axes[2, column].set_yticks(np.insert(probability_ticks, 1, 0.005))
        for tick_label in axes[2, column].get_yticklabels():
            if tick_label.get_text() == "0.005":
                tick_label.set_fontsize(9)
        if column == 2:
            axes[0, column].legend(
                loc="lower right",
                fontsize=13,
                framealpha=0.92,
                borderpad=0.35,
                labelspacing=0.3,
                handlelength=1.7,
                handletextpad=0.5,
            )
            axes[1, column].legend(
                loc="upper right",
                fontsize=15,
                framealpha=0.92,
            )
            axes[2, column].legend(
                loc="upper right",
                fontsize=15,
                framealpha=0.92,
            )
    axes[0, 0].set_ylabel("Coverage probability", fontsize=17)
    axes[1, 0].set_ylabel("Tail probability bounds", fontsize=17)
    axes[2, 0].set_ylabel("Interval width", fontsize=17)
    figure.subplots_adjust(
        left=0.085,
        right=0.99,
        top=0.95,
        bottom=0.085,
        wspace=0.10,
        hspace=0.10,
    )
    path = output_dir / "bounded_support.png"
    figure.savefig(path, dpi=300)
    plt.close(figure)
    return path


def _parse_scientific(value: str) -> float:
    match = re.fullmatch(r"([0-9.]+)\\times 10\^\{(-?\d+)\}", value)
    if not match:
        raise ValueError(value)
    return float(match.group(1)) * 10 ** int(match.group(2))


def _manuscript_values() -> list[dict[str, float]]:
    text = MANUSCRIPT_PATH.read_text(encoding="utf-8")
    start = text.index(r"\begin{table}[!ht]", text.index(r"\section{Further Numerical Details"))
    end = text.index(r"\label{tb_bounded_support}", start)
    block = text[start:end]
    quantile: float | None = None
    rows: list[dict[str, float]] = []
    quantile_pattern = re.compile(r"q_\{(0\.[0-9]+)\}")
    endpoint_pattern = re.compile(r"& \$(\\infty|[0-9]+\.[0-9]+)\$")
    coverage_pattern = re.compile(r"\$([0-9.]+) \(\\pm ([0-9.]+)\)\$")
    estimate_pattern = re.compile(
        r"\$([0-9.]+\\times 10\^\{-?\d+\})"
        r"\(\\pm ([0-9.]+\\times 10\^\{-?\d+\})\)\$"
    )
    for line in block.splitlines():
        quantile_match = quantile_pattern.search(line)
        if quantile_match:
            quantile = float(quantile_match.group(1))
        endpoint_match = endpoint_pattern.search(line)
        coverage_match = coverage_pattern.search(line)
        estimates = estimate_pattern.findall(line)
        if quantile is None or not endpoint_match or not coverage_match or len(estimates) != 3:
            continue
        endpoint_text = endpoint_match.group(1)
        values = [_parse_scientific(item) for pair in estimates for item in pair]
        rows.append(
            {
                "quantile": quantile,
                "endpoint": math.inf if endpoint_text == r"\infty" else float(endpoint_text),
                "coverage": float(coverage_match.group(1)),
                "coverage_half_width": float(coverage_match.group(2)),
                "lower": values[0],
                "lower_half_width": values[1],
                "upper": values[2],
                "upper_half_width": values[3],
                "width": values[4],
                "width_half_width": values[5],
            }
        )
    if len(rows) != 30:
        raise ValueError(f"Expected 30 manuscript rows, parsed {len(rows)}")
    return rows


def verify(raw_path: Path, output_dir: Path) -> Path:
    generated = aggregate(raw_path)
    expected = _manuscript_values()
    raw_rows = _read(raw_path)
    zero_intervals = sum(
        float(row["lower_bound"]) == 0.0 and float(row["upper_bound"]) == 0.0
        for row in raw_rows
    )
    fields = (
        "coverage",
        "coverage_half_width",
        "lower",
        "lower_half_width",
        "upper",
        "upper_half_width",
        "width",
        "width_half_width",
    )
    comparisons: list[tuple[float, float, str, float, float, float]] = []
    for actual, reference in zip(generated, expected):
        if (actual["quantile"], actual["endpoint"]) != (
            reference["quantile"],
            reference["endpoint"],
        ):
            raise ValueError("Generated and manuscript row order differs")
        for field in fields:
            comparisons.append(
                (
                    actual["quantile"],
                    actual["endpoint"],
                    field,
                    actual[field],
                    reference[field],
                    abs(actual[field] - reference[field]),
                )
            )
    exact_display = sum(
        round(actual, 2) == reference
        if field.startswith("coverage")
        else _scientific(actual) == _scientific(reference)
        for _, _, field, actual, reference, _ in comparisons
    )
    probability_comparisons = [
        item for item in comparisons if not item[2].startswith("coverage")
    ]
    coverage_comparisons = [
        item for item in comparisons if item[2].startswith("coverage")
    ]
    probability_within = sum(
        difference <= 1e-4 + 1e-12
        for _, _, _, _, _, difference in probability_comparisons
    )
    coverage_within = sum(
        difference <= 0.01 + 1e-12
        for _, _, _, _, _, difference in coverage_comparisons
    )
    behavior_correlations: dict[str, float] = {}
    for field in ("coverage", "lower", "upper", "width"):
        actual_values = np.asarray([row[field] for row in generated], dtype=float)
        reference_values = np.asarray([row[field] for row in expected], dtype=float)
        behavior_correlations[field] = float(
            np.corrcoef(actual_values, reference_values)[0, 1]
        )
    generated_below = [
        row["coverage"] for row in generated if row["endpoint"] < 2.0
    ]
    generated_valid = [
        row["coverage"] for row in generated if row["endpoint"] >= 2.0
    ]
    expected_below = [
        row["coverage"] for row in expected if row["endpoint"] < 2.0
    ]
    expected_valid = [
        row["coverage"] for row in expected if row["endpoint"] >= 2.0
    ]
    worst = sorted(comparisons, key=lambda item: item[-1], reverse=True)[:10]
    report = [
        "# Figure 2 and Table I.1 verification report",
        "",
        f"- Repetitions: {REPETITIONS} per displayed cell ({len(generated) * REPETITIONS} interval estimates)",
        f"- Underlying bound optimizations: {2 * len(generated) * REPETITIONS} (one minimization and one maximization per interval)",
        f"- Finite, ordered intervals: {len(raw_rows)}/{len(raw_rows)}",
        f"- Explicit `[0, 0]` solver-fallback intervals retained in aggregation: {zero_intervals}",
        "- Existing implementation reused: `estimate_tail_probability_D2_chi2_only`",
        "- Original seed and all twelve sampling groups retained; only the ten manuscript endpoints are solved and displayed",
        f"- Table fields matching manuscript display precision: {exact_display}/{len(comparisons)}",
        f"- Bound/width fields within `1e-4`: {probability_within}/{len(probability_comparisons)}; maximum difference `{max(item[-1] for item in probability_comparisons):.6g}`",
        f"- Coverage fields within `0.01`: {coverage_within}/{len(coverage_comparisons)}; maximum difference `{max(item[-1] for item in coverage_comparisons):.6g}`",
        "- Generated/manuscript correlations for the four behavioral series: "
        + ", ".join(
            f"`{field}` {correlation:.6f}"
            for field, correlation in behavior_correlations.items()
        ),
        f"- Mean coverage with misspecified endpoint below 2: generated `{np.mean(generated_below):.3f}`, manuscript `{np.mean(expected_below):.3f}`",
        f"- Mean coverage at or above the true endpoint 2: generated `{np.mean(generated_valid):.3f}`, manuscript `{np.mean(expected_valid):.3f}`",
        "- Figure criterion: substantive scientific/visual equivalence; PNG byte equality is not required",
        f"- Generated figure: `{output_dir / MANUSCRIPT_FIGURE.name}`",
        f"- Manuscript figure: `{MANUSCRIPT_FIGURE}`",
        "",
        "| Quantile | Endpoint | Field | Generated | Manuscript | Absolute difference |",
        "| ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for quantile, endpoint, field, actual, reference, difference in worst:
        endpoint_label = "inf" if math.isinf(endpoint) else f"{endpoint:g}"
        report.append(
            f"| {quantile:g} | {endpoint_label} | `{field}` | {actual:.8g} | {reference:.8g} | {difference:.3g} |"
        )
    path = output_dir / "verification_report.md"
    path.write_text("\n".join(report) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("generate", "render", "plot", "verify", "all"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raw_path = args.output_dir / "raw_results.csv"
    if args.stage in {"generate", "all"}:
        raw_path = generate(args.output_dir, args.workers)
    if args.stage in {"render", "all"}:
        render(raw_path, args.output_dir)
    if args.stage in {"plot", "all"}:
        plot(raw_path, args.output_dir)
    if args.stage == "verify":
        verify(raw_path, args.output_dir)


if __name__ == "__main__":
    main()
