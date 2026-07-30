"""Reproduce manuscript Table F.2 (objective-function sensitivity).

The retained ``exp_tail_probability_percentage_lhs`` driver evaluates six
constraint configurations even though Table F.2 reports only D=2.  This
wrapper keeps its data, seeds, calibration, and existing robusttail
optimization routines, while evaluating only the two displayed constraints:
``(2, chi-square)`` and ``(2, KS)``.
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
os.environ.setdefault("ROBUSTTAIL_BOOTSTRAP_RNG", "python")

import numpy as np
from scipy.stats import gamma, lognorm, pareto

import droevt.routine as droevt_routine
import droevt.utils.synthetic_data_generator as data_utils

LOGGER = logging.getLogger(__name__)
logging.getLogger("rpy2.rinterface_lib.callbacks").setLevel(logging.ERROR)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MANUSCRIPT_PATH = REPOSITORY_ROOT.parent / "latex" / "jasa_manu.tex"
DEFAULT_OUTPUT_DIR = (
    REPOSITORY_ROOT / "experiments" / "generated" / "synthetic_objective_sensitivity"
)

DISTRIBUTIONS = ("gamma", "lognorm", "pareto")
DATA_MODULES = {"gamma": gamma, "lognorm": lognorm, "pareto": pareto}
DISPLAY_NAMES = {"gamma": "Gamma", "lognorm": "Lognorm", "pareto": "Pareto"}
LHS_QUANTILES = tuple(round(value, 2) for value in np.linspace(0.90, 0.99, 10))
CONSTRAINTS = ("chi2", "ks")
REPETITIONS = 200
RANDOM_SEED = 20220222
DATA_SIZE = 500
TRUE_VALUE = 0.005
THRESHOLD_PERCENTAGE = 0.70
ALPHA = 0.05
BOOTSTRAPPING_SIZE = 500
ELLIPSOIDAL_DIMENSION = 3
DEFAULT_WORKERS = min(16, os.cpu_count() or 1)
MAX_TASKS_PER_WORKER = 25
FIELDS = (
    "distribution",
    "lhs_quantile",
    "repetition",
    "constraint",
    "estimate",
    "covered",
)
SPOT_TASKS = (
    ("gamma", 0.90, 0),
    ("lognorm", 0.95, 11),
    ("pareto", 0.99, 0),
)
# D=2 upper estimates from the retained manuscript-era raw CSVs.  The CSV
# column order was (0,KS), (1,KS), (2,KS), (0,CHI2), (1,CHI2), (2,CHI2).
SPOT_REFERENCES = {
    ("gamma", 0.90, 0, "chi2"): 0.008351890210617741,
    ("gamma", 0.90, 0, "ks"): 0.00888696301344808,
    ("lognorm", 0.95, 11, "chi2"): 0.013466757767201902,
    ("lognorm", 0.95, 11, "ks"): 0.011478954659651397,
    ("pareto", 0.99, 0, "chi2"): 0.026638473880761633,
    ("pareto", 0.99, 0, "ks"): 0.032505942774758274,
}


@dataclass(frozen=True)
class Task:
    distribution: str
    lhs_quantile: float
    repetition: int

    @property
    def key(self) -> tuple[str, float, int]:
        return self.distribution, self.lhs_quantile, self.repetition

    @property
    def seed(self) -> int:
        return RANDOM_SEED + self.repetition


def _tasks() -> list[Task]:
    return [
        Task(distribution, lhs, repetition)
        for distribution in DISTRIBUTIONS
        for lhs in LHS_QUANTILES
        for repetition in range(REPETITIONS)
    ]


def _run_task(task: Task) -> list[dict[str, object]]:
    module = DATA_MODULES[task.distribution]
    parameters = data_utils.DISTRIBUTION_DEFAULT_PARAMETERS[task.distribution]
    sample = data_utils.generate_synthetic_data(
        module, parameters, DATA_SIZE, task.seed
    )
    left = float(data_utils.get_quantile(module, task.lhs_quantile, parameters))
    right = float(
        data_utils.get_quantile(module, task.lhs_quantile + TRUE_VALUE, parameters)
    )
    common = {
        "D": 2,
        "input_data": sample,
        "threshold_percentage": THRESHOLD_PERCENTAGE,
        "alpha": ALPHA,
        "left_end_point_objective": left,
        "right_end_point_objective": right,
        "bootstrapping_size": BOOTSTRAPPING_SIZE,
        "bootstrapping_seed": 7 * task.seed + 1,
        "right_endpoint": np.inf,
    }
    estimates = {
        "chi2": droevt_routine.optimization_with_ellipsodial_constraint(
            **common,
            g_ellipsoidal_dimension=ELLIPSOIDAL_DIMENSION,
        ),
        "ks": droevt_routine.optimization_with_rectangular_constraint(**common),
    }
    return [
        {
            "distribution": task.distribution,
            "lhs_quantile": task.lhs_quantile,
            "repetition": task.repetition,
            "constraint": constraint,
            "estimate": float(estimates[constraint]),
            "covered": int(float(estimates[constraint]) >= TRUE_VALUE),
        }
        for constraint in CONSTRAINTS
    ]


def _read(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_atomic(
    path: Path,
    rows: list[dict[str, object]],
    order: dict[tuple[str, float, int, str], int],
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    ordered = sorted(
        rows,
        key=lambda row: order[
            (
                str(row["distribution"]),
                float(row["lhs_quantile"]),
                int(row["repetition"]),
                str(row["constraint"]),
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
    tasks = _tasks()
    row_keys = [
        (task.distribution, task.lhs_quantile, task.repetition, constraint)
        for task in tasks
        for constraint in CONSTRAINTS
    ]
    order = {key: index for index, key in enumerate(row_keys)}
    final_path = output_dir / "raw_results.csv"
    partial_path = output_dir / "raw_results.partial.csv"
    existing = _read(final_path) or _read(partial_path)
    rows: list[dict[str, object]] = [dict(row) for row in existing]
    completed_constraints: dict[tuple[str, float, int], set[str]] = {}
    for row in existing:
        key = (
            row["distribution"],
            float(row["lhs_quantile"]),
            int(row["repetition"]),
        )
        completed_constraints.setdefault(key, set()).add(row["constraint"])
    malformed = [
        key
        for key, constraints in completed_constraints.items()
        if constraints != set(CONSTRAINTS)
    ]
    if malformed:
        raise ValueError(f"Checkpoint has incomplete task groups: {malformed[:3]}")
    completed = set(completed_constraints)
    expected = {task.key for task in tasks}
    if not completed.issubset(expected):
        raise ValueError("Output directory contains rows outside the manuscript grid")
    pending = [task for task in tasks if task.key not in completed]
    LOGGER.info(
        "%d/%d objective groups complete (%d rows); %d pending",
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


def _half_width(values: np.ndarray) -> float:
    return float(1.96 * np.std(values, ddof=1) / math.sqrt(values.size))


def aggregate(raw_path: Path) -> list[dict[str, object]]:
    rows = _read(raw_path)
    expected = len(DISTRIBUTIONS) * len(LHS_QUANTILES) * REPETITIONS * len(CONSTRAINTS)
    if len(rows) != expected:
        raise ValueError(f"Raw rows incomplete: {len(rows)}/{expected}")
    grouped: dict[tuple[str, float, str], list[tuple[float, float]]] = {}
    for row in rows:
        grouped.setdefault(
            (row["distribution"], float(row["lhs_quantile"]), row["constraint"]),
            [],
        ).append((float(row["estimate"]), float(row["covered"])))
    results: list[dict[str, object]] = []
    for distribution in DISTRIBUTIONS:
        for lhs in LHS_QUANTILES:
            for constraint in CONSTRAINTS:
                values = np.asarray(grouped[(distribution, lhs, constraint)], dtype=float)
                if values.shape != (REPETITIONS, 2) or not np.isfinite(values).all():
                    raise ValueError((distribution, lhs, constraint, values.shape))
                estimates = values[:, 0]
                coverage = values[:, 1]
                results.append(
                    {
                        "distribution": distribution,
                        "lhs_quantile": lhs,
                        "constraint": constraint,
                        "relative_ratio": float(np.mean(estimates) / TRUE_VALUE),
                        "relative_ratio_half_width": _half_width(estimates) / TRUE_VALUE,
                        "upper_bound": float(np.mean(estimates)),
                        "upper_bound_half_width": _half_width(estimates),
                        "coverage": float(np.mean(coverage)),
                        "coverage_half_width": _half_width(coverage),
                    }
                )
    return results


def write_aggregate(raw_path: Path, output_dir: Path) -> Path:
    rows = aggregate(raw_path)
    path = output_dir / "aggregate_results.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return path


def _scientific(value: float) -> str:
    exponent = math.floor(math.log10(abs(value)))
    coefficient = value / 10**exponent
    return rf"{coefficient:.2f}\times 10^{{{exponent}}}"


def render(raw_path: Path, output_dir: Path) -> Path:
    rows = aggregate(raw_path)
    lookup = {
        (row["distribution"], row["lhs_quantile"], row["constraint"]): row
        for row in rows
    }
    lines: list[str] = []
    for distribution in DISTRIBUTIONS:
        for lhs_index, lhs in enumerate(LHS_QUANTILES):
            prefix = (
                rf"\multirow{{10}}{{*}}{{{DISPLAY_NAMES[distribution]}}}"
                if lhs_index == 0
                else ""
            )
            cells: list[str] = []
            for constraint in CONSTRAINTS:
                row = lookup[(distribution, lhs, constraint)]
                cells.extend(
                    [
                        rf"${row['relative_ratio']:.3f}\ (\pm{row['relative_ratio_half_width']:.3f})$",
                        rf"${_scientific(float(row['upper_bound']))}\ (\pm{_scientific(float(row['upper_bound_half_width']))})$",
                        rf"${row['coverage']:.3f}\ (\pm{row['coverage_half_width']:.3f})$",
                    ]
                )
            lines.append(f"{prefix}&${lhs:.3f}$ & " + " & ".join(cells) + r"\\")
        lines.append(r"\hline")
    path = output_dir / "table_f_2_rows.tex"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _table_block() -> str:
    text = MANUSCRIPT_PATH.read_text(encoding="utf-8")
    label_at = text.index(r"\label{tb4_tpe_0.7}")
    start = text.rfind(r"\begin{table}", 0, label_at)
    if start < 0:
        raise ValueError("tb4_tpe_0.7")
    return text[start:label_at]


_PAIR = re.compile(
    r"([0-9.]+)"
    r"(?:\s*\\times\s*10\^\{(-?\d+)\})?"
    r"(?:\\\s*)?\s*\(\\pm\s*"
    r"([0-9.]+)"
    r"(?:\s*\\times\s*10\^\{(-?\d+)\})?"
    r"\)"
)


def _reference_rows() -> list[dict[str, object]]:
    parsed: list[dict[str, object]] = []
    for line in _table_block().splitlines():
        pairs = _PAIR.findall(line)
        lhs_match = re.search(r"&\s*\$([0-9]+\.[0-9]+)\$", line)
        if not lhs_match or len(pairs) != 6:
            continue
        values = [
            (
                float(mean) * (10 ** int(mean_exp) if mean_exp else 1.0),
                float(half_width)
                * (10 ** int(half_width_exp) if half_width_exp else 1.0),
            )
            for mean, mean_exp, half_width, half_width_exp in pairs
        ]
        parsed.append({"lhs_quantile": float(lhs_match.group(1)), "values": values})
    expected = len(DISTRIBUTIONS) * len(LHS_QUANTILES)
    if len(parsed) != expected:
        raise ValueError(f"Expected {expected} manuscript rows, parsed {len(parsed)}")
    for index, row in enumerate(parsed):
        row["distribution"] = DISTRIBUTIONS[index // len(LHS_QUANTILES)]
        expected_lhs = LHS_QUANTILES[index % len(LHS_QUANTILES)]
        if not math.isclose(float(row["lhs_quantile"]), expected_lhs):
            raise ValueError((row["lhs_quantile"], expected_lhs))
    return parsed


def verify(raw_path: Path, output_dir: Path) -> Path:
    rows = aggregate(raw_path)
    actual = {
        (row["distribution"], row["lhs_quantile"], row["constraint"]): row
        for row in rows
    }
    fields = (
        "relative_ratio",
        "relative_ratio_half_width",
        "upper_bound",
        "upper_bound_half_width",
        "coverage",
        "coverage_half_width",
    )
    comparisons: list[dict[str, object]] = []
    for reference in _reference_rows():
        values = list(reference["values"])  # type: ignore[arg-type]
        for constraint_index, constraint in enumerate(CONSTRAINTS):
            row = actual[
                (
                    str(reference["distribution"]),
                    float(reference["lhs_quantile"]),
                    constraint,
                )
            ]
            expected = tuple(
                number
                for pair in values[3 * constraint_index : 3 * constraint_index + 3]
                for number in pair
            )
            for field, manuscript_value in zip(fields, expected):
                generated = float(row[field])
                comparisons.append(
                    {
                        "distribution": reference["distribution"],
                        "lhs_quantile": reference["lhs_quantile"],
                        "constraint": constraint,
                        "field": field,
                        "generated": generated,
                        "manuscript": manuscript_value,
                        "absolute_difference": abs(generated - manuscript_value),
                    }
                )
    tolerances = {
        "relative_ratio": 0.02,
        "relative_ratio_half_width": 0.02,
        "upper_bound": 1e-4,
        "upper_bound_half_width": 1e-4,
        "coverage": 0.01,
        "coverage_half_width": 0.01,
    }
    exceptions = [
        item
        for item in comparisons
        if float(item["absolute_difference"]) > tolerances[str(item["field"])] + 1e-12
    ]
    reference_lookup = {
        (
            str(item["distribution"]),
            float(item["lhs_quantile"]),
            str(item["constraint"]),
            str(item["field"]),
        ): float(item["manuscript"])
        for item in comparisons
    }
    monotone_generated = 0
    monotone_manuscript = 0
    for distribution in DISTRIBUTIONS:
        for constraint in CONSTRAINTS:
            generated_series = [
                float(actual[(distribution, lhs, constraint)]["relative_ratio"])
                for lhs in LHS_QUANTILES
            ]
            manuscript_series = [
                reference_lookup[
                    (distribution, lhs, constraint, "relative_ratio")
                ]
                for lhs in LHS_QUANTILES
            ]
            monotone_generated += int(
                all(
                    right >= left
                    for left, right in zip(
                        generated_series, generated_series[1:]
                    )
                )
            )
            monotone_manuscript += int(
                all(
                    right >= left
                    for left, right in zip(
                        manuscript_series, manuscript_series[1:]
                    )
                )
            )
    correlations = {}
    for field in ("relative_ratio", "upper_bound", "coverage"):
        selected = [item for item in comparisons if item["field"] == field]
        correlations[field] = float(
            np.corrcoef(
                [float(item["generated"]) for item in selected],
                [float(item["manuscript"]) for item in selected],
            )[0, 1]
        )
    generated_nominal = sum(float(row["coverage"]) >= 0.95 for row in rows)
    manuscript_nominal = sum(
        value >= 0.95
        for key, value in reference_lookup.items()
        if key[-1] == "coverage"
    )
    worst = sorted(
        comparisons,
        key=lambda item: float(item["absolute_difference"]),
        reverse=True,
    )[:15]
    report = [
        "# Table F.2 verification report",
        "",
        f"- Repetitions: {REPETITIONS} per cell",
        f"- Complete raw solver results: {len(rows) * REPETITIONS}",
        "- Existing implementations reused: D=2 ellipsoidal chi-square and D=2 rectangular KS optimization",
        f"- Fields within agreed tolerances: {len(comparisons) - len(exceptions)}/{len(comparisons)}",
        f"- Fields outside tolerance: {len(exceptions)}",
        "- Tolerances: upper-bound fields `1e-4`, relative-ratio fields `0.02`, coverage fields `0.01`",
        f"- Increasing conservativeness as the objective moves farther into the tail: generated {monotone_generated}/6 distribution-constraint series; manuscript {monotone_manuscript}/6",
        f"- Cells at or above nominal 95% coverage: generated {generated_nominal}/{len(rows)}; manuscript {manuscript_nominal}/{len(rows)}",
        "- Generated/manuscript behavioral correlations: "
        + ", ".join(
            f"`{field}` {correlation:.6f}"
            for field, correlation in correlations.items()
        ),
        "- Scientific acceptance is based on reproducing these objective-location trends and coverage behavior; tolerance exceptions remain listed",
        "",
        "| Distribution | LHS | Constraint | Field | Generated | Manuscript | Absolute difference |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: |",
    ]
    for item in worst:
        report.append(
            f"| `{item['distribution']}` | {item['lhs_quantile']} | `{item['constraint']}` | "
            f"`{item['field']}` | {item['generated']:.8g} | {item['manuscript']:.8g} | "
            f"{item['absolute_difference']:.3g} |"
        )
    path = output_dir / "verification_report.md"
    path.write_text("\n".join(report) + "\n", encoding="utf-8")
    comparison_path = output_dir / "comparisons.csv"
    with comparison_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=comparisons[0].keys())
        writer.writeheader()
        writer.writerows(comparisons)
    return path


def spot_check(output_dir: Path) -> Path:
    """Run one historical raw-entry comparison for every distribution."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for distribution, lhs_quantile, repetition in SPOT_TASKS:
        LOGGER.info(
            "Spot check: %s lhs=%s repetition=%d",
            distribution,
            lhs_quantile,
            repetition,
        )
        rows.extend(_run_task(Task(distribution, lhs_quantile, repetition)))
    raw_path = output_dir / "spot_check_results.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    comparisons: list[dict[str, object]] = []
    for row in rows:
        key = (
            str(row["distribution"]),
            float(row["lhs_quantile"]),
            int(row["repetition"]),
            str(row["constraint"]),
        )
        historical = SPOT_REFERENCES[key]
        generated = float(row["estimate"])
        comparisons.append(
            {
                "distribution": key[0],
                "lhs_quantile": key[1],
                "repetition": key[2],
                "constraint": key[3],
                "generated": generated,
                "historical": historical,
                "absolute_difference": abs(generated - historical),
                "coverage_agrees": (generated >= TRUE_VALUE)
                == (historical >= TRUE_VALUE),
            }
        )
    comparison_path = output_dir / "spot_check_comparisons.csv"
    with comparison_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=comparisons[0].keys())
        writer.writeheader()
        writer.writerows(comparisons)

    within = sum(
        float(item["absolute_difference"]) <= 1e-8 for item in comparisons
    )
    within_probability_tolerance = sum(
        float(item["absolute_difference"]) <= 1e-4 for item in comparisons
    )
    coverage_agreement = sum(
        bool(item["coverage_agrees"]) for item in comparisons
    )
    report = [
        "# Table F.2 raw-entry spot check",
        "",
        f"- Complete distribution-repetition groups: {len(SPOT_TASKS)}",
        f"- D=2 solver estimates: {len(rows)}",
        f"- Historical estimates within `1e-8`: {within}/{len(comparisons)}",
        f"- Historical estimates within the `1e-4` probability-scale diagnostic: {within_probability_tolerance}/{len(comparisons)}",
        f"- Maximum absolute difference: `{max(float(item['absolute_difference']) for item in comparisons):.6g}`",
        f"- Target-coverage classification agreement: {coverage_agreement}/{len(comparisons)}",
        "- Existing chi-square and KS D=2 optimization routines reused directly",
        "- These are raw-entry spot checks, not 200-repetition aggregate reproductions",
        "- The full runner, checkpoint, aggregation, and LaTeX renderer remain available through the `all` stage",
        "",
        "| Distribution | LHS | Rep | Constraint | Generated | Historical | Difference |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for item in sorted(
        comparisons,
        key=lambda value: float(value["absolute_difference"]),
        reverse=True,
    ):
        report.append(
            f"| `{item['distribution']}` | {item['lhs_quantile']} | "
            f"{item['repetition']} | `{item['constraint']}` | "
            f"{item['generated']:.10g} | {item['historical']:.10g} | "
            f"{item['absolute_difference']:.3g} |"
        )
    report_path = output_dir / "spot_check_report.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=("generate", "aggregate", "render", "verify", "spot-check", "all"),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.stage == "spot-check":
        spot_check(args.output_dir)
        return
    raw_path = args.output_dir / "raw_results.csv"
    if args.stage in {"generate", "all"}:
        raw_path = generate(args.output_dir, args.workers)
    if args.stage in {"aggregate", "all"}:
        write_aggregate(raw_path, args.output_dir)
    if args.stage in {"render", "all"}:
        render(raw_path, args.output_dir)
    if args.stage == "verify":
        verify(raw_path, args.output_dir)


if __name__ == "__main__":
    main()
