"""Reproduce Table F.1(b), quantile estimation across shape constraints.

The wrapper delegates each simulated dataset to the retained
``quantileEstimationPerRep`` implementation, which generates one deterministic
sample and evaluates the three displayed chi-square configurations D=0, 1,
and 2.  It adds manuscript-scale repetition counts, parallel execution,
checkpointing, aggregation, LaTeX rendering, and numerical/behavioral
comparison.
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

import droevt.utils.synthetic_data_generator as data_utils
from experiments.run_scripts.quantile_estimation.quantileEstimationUnit import (
    quantileEstimationPerRep,
)

LOGGER = logging.getLogger(__name__)
logging.getLogger("rpy2.rinterface_lib.callbacks").setLevel(logging.ERROR)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MANUSCRIPT_PATH = REPOSITORY_ROOT.parent / "latex" / "jasa_manu.tex"
DEFAULT_OUTPUT_DIR = (
    REPOSITORY_ROOT
    / "experiments"
    / "generated"
    / "synthetic_quantile_constraint_comparison"
)

DISTRIBUTIONS = ("gamma", "lognorm", "pareto")
DATA_MODULES = {"gamma": gamma, "lognorm": lognorm, "pareto": pareto}
DISPLAY_NAMES = {"gamma": "Gamma", "lognorm": "Lognorm", "pareto": "Pareto"}
CONSTRAINTS = ("chi2_d0", "chi2_d1", "chi2_d2")
CONSTRAINT_LATEX = {
    "chi2_d0": r"(0,\chi^2)",
    "chi2_d1": r"(1,\chi^2)",
    "chi2_d2": r"(2,\chi^2)",
}
REPETITIONS = 200
RANDOM_SEED = 20220222
DATA_SIZE = 500
QUANTILE_LEVEL = 0.99
THRESHOLD_PERCENTAGE = 0.70
ALPHA = 0.05
ELLIPSOIDAL_DIMENSION = 3
DEFAULT_WORKERS = min(16, os.cpu_count() or 1)
MAX_TASKS_PER_WORKER = 25
FIELDS = (
    "distribution",
    "repetition",
    "random_seed",
    *CONSTRAINTS,
)


@dataclass(frozen=True)
class Task:
    distribution: str
    repetition: int

    @property
    def seed(self) -> int:
        return RANDOM_SEED + self.repetition

    @property
    def key(self) -> tuple[str, int]:
        return self.distribution, self.repetition


SPOT_TASKS = tuple(Task(distribution, 0) for distribution in DISTRIBUTIONS)
SPOT_REFERENCES = {
    ("gamma", 0): (
        7.20765712462264,
        5.024211127936701,
        4.6547229285605,
    ),
    ("lognorm", 0): (
        19.118227484897453,
        13.683575398812696,
        12.834733718419223,
    ),
    ("pareto", 0): (
        89.96314176254536,
        62.951957510961805,
        58.83806846616018,
    ),
}


def _tasks() -> list[Task]:
    return [
        Task(distribution, repetition)
        for distribution in DISTRIBUTIONS
        for repetition in range(REPETITIONS)
    ]


def _run_task(task: Task) -> dict[str, object]:
    estimates = quantileEstimationPerRep(
        dataModule=DATA_MODULES[task.distribution],
        quantitleValue=QUANTILE_LEVEL,
        dataSize=DATA_SIZE,
        thresholdPercentage=THRESHOLD_PERCENTAGE,
        gEllipsoidalDimension=ELLIPSOIDAL_DIMENSION,
        alpha=ALPHA,
        random_state=task.seed,
    )
    if len(estimates) != len(CONSTRAINTS):
        raise ValueError(f"Expected three chi-square estimates, received {estimates}")
    return {
        "distribution": task.distribution,
        "repetition": task.repetition,
        "random_seed": task.seed,
        **{
            constraint: float(estimates[index])
            for index, constraint in enumerate(CONSTRAINTS)
        },
    }


def _read(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_atomic(
    path: Path,
    rows: list[dict[str, object]],
    order: dict[tuple[str, int], int],
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    ordered = sorted(
        rows,
        key=lambda row: order[(str(row["distribution"]), int(row["repetition"]))],
    )
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(ordered)
    temporary.replace(path)


def generate(output_dir: Path, workers: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = _tasks()
    order = {task.key: index for index, task in enumerate(tasks)}
    final_path = output_dir / "raw_results.csv"
    partial_path = output_dir / "raw_results.partial.csv"
    existing = _read(final_path) or _read(partial_path)
    rows: list[dict[str, object]] = [dict(row) for row in existing]
    completed = {
        (row["distribution"], int(row["repetition"])) for row in existing
    }
    if len(completed) != len(existing) or not completed.issubset(order):
        raise ValueError("Checkpoint has duplicate or unexpected tasks")
    pending = [task for task in tasks if task.key not in completed]
    LOGGER.info(
        "%d/%d replicate groups complete; %d pending",
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
        for completed_now, row in enumerate(
            pool.imap_unordered(_run_task, pending, chunksize=1), start=1
        ):
            rows.append(row)
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
    if len(rows) != len(DISTRIBUTIONS) * REPETITIONS:
        raise ValueError(
            f"Raw replicate groups incomplete: {len(rows)}/"
            f"{len(DISTRIBUTIONS) * REPETITIONS}"
        )
    results: list[dict[str, object]] = []
    for distribution in DISTRIBUTIONS:
        selected = [row for row in rows if row["distribution"] == distribution]
        if len(selected) != REPETITIONS:
            raise ValueError((distribution, len(selected)))
        parameters = data_utils.DISTRIBUTION_DEFAULT_PARAMETERS[distribution]
        true_value = float(
            data_utils.get_quantile(
                DATA_MODULES[distribution], QUANTILE_LEVEL, parameters
            )
        )
        for constraint in CONSTRAINTS:
            estimates = np.asarray(
                [float(row[constraint]) for row in selected], dtype=float
            )
            if estimates.shape != (REPETITIONS,) or not np.isfinite(estimates).all():
                raise ValueError((distribution, constraint, estimates.shape))
            coverage = (estimates >= true_value).astype(float)
            results.append(
                {
                    "distribution": distribution,
                    "constraint": constraint,
                    "true_value": true_value,
                    "relative_ratio": float(np.mean(estimates) / true_value),
                    "relative_ratio_half_width": _half_width(estimates) / true_value,
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
    mantissa, exponent = f"{value:.2E}".split("E")
    return rf"{mantissa}\times 10^{{{int(exponent)}}}"


def render(raw_path: Path, output_dir: Path) -> Path:
    rows = aggregate(raw_path)
    lookup = {(row["distribution"], row["constraint"]): row for row in rows}
    lines: list[str] = []
    for distribution in DISTRIBUTIONS:
        for constraint_index, constraint in enumerate(CONSTRAINTS):
            row = lookup[(distribution, constraint)]
            prefix = (
                rf"\multirow{{3}}{{*}}{{{DISPLAY_NAMES[distribution]}}}"
                if constraint_index == 0
                else ""
            )
            ending = r"\\\hline " if constraint_index == 2 else r"\\"
            lines.append(
                f"{prefix}&${CONSTRAINT_LATEX[constraint]}$& "
                f"${row['relative_ratio']:.2f}\\ (\\pm{row['relative_ratio_half_width']:.2f})$ & "
                f"${_scientific(float(row['upper_bound']))}\\ (\\pm {_scientific(float(row['upper_bound_half_width']))})$ & "
                f"${row['coverage']:.3f}\\ (\\pm {row['coverage_half_width']:.3f})$"
                f"{ending}"
            )
    path = output_dir / "table_f_1b_rows.tex"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _active_subtable() -> str:
    text = MANUSCRIPT_PATH.read_text(encoding="utf-8")
    matches = list(
        re.finditer(
            r"^[ \t]*\\label\{tb2_qe\}[ \t]*$",
            text,
            flags=re.MULTILINE,
        )
    )
    if len(matches) != 1:
        raise ValueError(f"Expected one active tb2_qe label, found {len(matches)}")
    start = text.rfind(r"\begin{subtable}", 0, matches[0].start())
    end = text.index(r"\end{subtable}", matches[0].end())
    return text[start:end]


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
    for line in _active_subtable().splitlines():
        pairs = _PAIR.findall(line)
        if len(pairs) != 3:
            continue
        values = [
            (
                float(mean) * (10 ** int(mean_exp) if mean_exp else 1.0),
                float(half_width)
                * (10 ** int(half_width_exp) if half_width_exp else 1.0),
            )
            for mean, mean_exp, half_width, half_width_exp in pairs
        ]
        parsed.append({"values": values})
    expected = len(DISTRIBUTIONS) * len(CONSTRAINTS)
    if len(parsed) != expected:
        raise ValueError(f"Expected {expected} manuscript rows, parsed {len(parsed)}")
    for index, row in enumerate(parsed):
        row["distribution"] = DISTRIBUTIONS[index // len(CONSTRAINTS)]
        row["constraint"] = CONSTRAINTS[index % len(CONSTRAINTS)]
    return parsed


def verify(raw_path: Path, output_dir: Path) -> Path:
    rows = aggregate(raw_path)
    actual = {
        (row["distribution"], row["constraint"]): row for row in rows
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
        row = actual[(reference["distribution"], reference["constraint"])]
        expected_values = tuple(
            number
            for pair in list(reference["values"])  # type: ignore[arg-type]
            for number in pair
        )
        for field, manuscript_value in zip(fields, expected_values):
            generated = float(row[field])
            comparisons.append(
                {
                    "distribution": reference["distribution"],
                    "constraint": reference["constraint"],
                    "field": field,
                    "generated": generated,
                    "manuscript": manuscript_value,
                    "absolute_difference": abs(generated - manuscript_value),
                    "relative_difference": (
                        abs(generated - manuscript_value) / abs(manuscript_value)
                        if manuscript_value
                        else abs(generated - manuscript_value)
                    ),
                }
            )
    monotone = 0
    for distribution in DISTRIBUTIONS:
        estimates = [
            float(actual[(distribution, constraint)]["upper_bound"])
            for constraint in CONSTRAINTS
        ]
        monotone += int(estimates[0] >= estimates[1] >= estimates[2])
    coverage_valid = sum(float(row["coverage"]) >= 0.95 for row in rows)
    worst = sorted(
        comparisons,
        key=lambda item: float(item["relative_difference"]),
        reverse=True,
    )[:15]
    report = [
        "# Table F.1(b) verification report",
        "",
        f"- Repetitions: {REPETITIONS} per distribution",
        f"- Complete quantile inversions: {len(rows) * REPETITIONS}",
        "- Existing implementation reused: `quantileEstimationPerRep` for D=0, D=1, and D=2 chi-square constraints",
        f"- Stronger constraints reproduce decreasing upper bounds: {monotone}/{len(DISTRIBUTIONS)} distributions",
        f"- Generated cells with at least nominal 95% coverage: {coverage_valid}/{len(rows)}",
        "- Acceptance emphasizes the manuscript behavior (stronger shape constraints tighten bounds while retaining coverage); all numerical deviations are reported below",
        "",
        "| Distribution | Constraint | Field | Generated | Manuscript | Absolute difference | Relative difference |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for item in worst:
        report.append(
            f"| `{item['distribution']}` | `{item['constraint']}` | `{item['field']}` | "
            f"{item['generated']:.8g} | {item['manuscript']:.8g} | "
            f"{item['absolute_difference']:.3g} | {item['relative_difference']:.3g} |"
        )
    path = output_dir / "verification_report.md"
    path.write_text("\n".join(report) + "\n", encoding="utf-8")
    comparison_path = output_dir / "comparisons.csv"
    with comparison_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=comparisons[0].keys())
        writer.writeheader()
        writer.writerows(comparisons)
    return path


def spot_check(output_dir: Path, workers: int) -> Path:
    """Compare one complete historical quantile-inversion group per distribution."""
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info(
        "Spot checking %d complete groups with %d workers",
        len(SPOT_TASKS),
        min(workers, len(SPOT_TASKS)),
    )
    context = multiprocessing.get_context("spawn")
    with context.Pool(
        processes=min(workers, len(SPOT_TASKS)),
        maxtasksperchild=1,
    ) as pool:
        rows = list(pool.imap(_run_task, SPOT_TASKS, chunksize=1))
    raw_path = output_dir / "spot_check_results.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    comparisons: list[dict[str, object]] = []
    for row in rows:
        key = (str(row["distribution"]), int(row["repetition"]))
        true_value = float(
            data_utils.get_quantile(
                DATA_MODULES[key[0]],
                QUANTILE_LEVEL,
                data_utils.DISTRIBUTION_DEFAULT_PARAMETERS[key[0]],
            )
        )
        for index, constraint in enumerate(CONSTRAINTS):
            generated = float(row[constraint])
            historical = SPOT_REFERENCES[key][index]
            comparisons.append(
                {
                    "distribution": key[0],
                    "repetition": key[1],
                    "constraint": constraint,
                    "generated": generated,
                    "historical": historical,
                    "absolute_difference": abs(generated - historical),
                    "coverage_agrees": (generated >= true_value)
                    == (historical >= true_value),
                }
            )
    comparison_path = output_dir / "spot_check_comparisons.csv"
    with comparison_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=comparisons[0].keys())
        writer.writeheader()
        writer.writerows(comparisons)

    exact = sum(
        float(item["absolute_difference"]) <= 1e-12 for item in comparisons
    )
    coverage = sum(bool(item["coverage_agrees"]) for item in comparisons)
    monotone = sum(
        float(row["chi2_d0"])
        >= float(row["chi2_d1"])
        >= float(row["chi2_d2"])
        for row in rows
    )
    report = [
        "# Table F.1(b) raw-entry spot check",
        "",
        f"- Complete distribution-repetition groups: {len(rows)}",
        f"- Quantile inversions: {len(comparisons)}",
        f"- Historical estimates within `1e-12`: {exact}/{len(comparisons)}",
        f"- Maximum absolute difference: `{max(float(item['absolute_difference']) for item in comparisons):.6g}`",
        f"- Target-coverage classification agreement: {coverage}/{len(comparisons)}",
        f"- D=0 >= D=1 >= D=2 ordering: {monotone}/{len(rows)} groups",
        "- Existing `quantileEstimationPerRep` implementation reused directly",
        "- These are raw-entry spot checks, not 200-repetition aggregate reproductions",
        "- The full runner, checkpoint, aggregation, LaTeX renderer, and manuscript comparator remain available through the `all` stage",
        "",
        "| Distribution | Rep | Constraint | Generated | Historical | Difference |",
        "| --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for item in comparisons:
        report.append(
            f"| `{item['distribution']}` | {item['repetition']} | "
            f"`{item['constraint']}` | {item['generated']:.12g} | "
            f"{item['historical']:.12g} | {item['absolute_difference']:.3g} |"
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
        spot_check(args.output_dir, args.workers)
        return
    raw_path = args.output_dir / "raw_results.csv"
    if args.stage in {"generate", "all"}:
        raw_path = generate(args.output_dir, args.workers)
    if args.stage in {"aggregate", "all"}:
        write_aggregate(raw_path, args.output_dir)
    if args.stage in {"render", "all"}:
        render(raw_path, args.output_dir)
    if args.stage in {"verify", "all"}:
        verify(raw_path, args.output_dir)


if __name__ == "__main__":
    main()
