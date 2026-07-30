"""Reproduce manuscript Table F.1(a), the six-constraint simulation study.

The implementation delegates each simulated dataset to the established
``estimate_tail_probability_with_data_module`` routine, which returns all
three KS and all three chi-square constraint results in one call.
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

ENVIRONMENT_PREFIX = Path(sys.executable).resolve().parent.parent
os.environ.setdefault("R_HOME", str(ENVIRONMENT_PREFIX / "lib" / "R"))
# Table F.1 predates the May 2025 NumPy bootstrap refactor.
os.environ.setdefault("ROBUSTTAIL_BOOTSTRAP_RNG", "python")

import numpy as np
import pandas as pd
from scipy.stats import gamma, lognorm, pareto

import droevt.utils.synthetic_data_generator as data_utils
from experiments.run_scripts.tail_probability.tail_probability_estimation import (
    estimate_tail_probability_with_data_module,
)

LOGGER = logging.getLogger(__name__)
logging.getLogger("rpy2.rinterface_lib.callbacks").setLevel(logging.ERROR)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = (
    REPOSITORY_ROOT / "experiments" / "generated" / "synthetic_constraint_comparison"
)
DEFAULT_MANUSCRIPT = REPOSITORY_ROOT.parent / "latex" / "jasa_manu.tex"

RANDOM_SEED = 20220222
DATA_SIZE = 500
REPETITIONS = 200
THRESHOLD = 0.70
PERCENTAGE_LHS = 0.99
PERCENTAGE_RHS = 0.995
TRUE_VALUE = PERCENTAGE_RHS - PERCENTAGE_LHS
ALPHA = 0.05
ELLIPSOIDAL_DIMENSION = 3
BOOTSTRAPPING_SIZE = 500
MONTE_CARLO_Z = 1.96
DEFAULT_WORKERS = min(16, os.cpu_count() or 1)
MAX_TASKS_PER_WORKER = 10

DISTRIBUTIONS = {"gamma": gamma, "lognorm": lognorm, "pareto": pareto}
DISPLAY_NAMES = {"gamma": "Gamma", "lognorm": "Lognorm", "pareto": "Pareto"}
# Existing routine order is KS D=0,1,2 followed by chi-square D=0,1,2.
CONSTRAINTS = (
    ("ks_d0", r"(0,\text{{KS}})", 0),
    ("ks_d1", r"(1,\text{{KS}})", 1),
    ("ks_d2", r"(2,\text{{KS}})", 2),
    ("chi2_d0", r"(0,\chi^2)", 3),
    ("chi2_d1", r"(1,\chi^2)", 4),
    ("chi2_d2", r"(2,\chi^2)", 5),
)
MANUSCRIPT_ORDER = ("chi2_d0", "chi2_d1", "chi2_d2", "ks_d0", "ks_d1", "ks_d2")
RAW_COLUMNS = (
    "distribution",
    "repetition",
    "random_seed",
    *(constraint[0] for constraint in CONSTRAINTS),
)


@dataclass(frozen=True)
class Task:
    distribution: str
    repetition: int

    @property
    def random_seed(self) -> int:
        return RANDOM_SEED + self.repetition

    @property
    def key(self) -> tuple[str, int]:
        return self.distribution, self.repetition


def _run_task(task: Task) -> dict[str, object]:
    estimates = estimate_tail_probability_with_data_module(
        data_module=DISTRIBUTIONS[task.distribution],
        percentage_lhs=PERCENTAGE_LHS,
        percentage_rhs=PERCENTAGE_RHS,
        data_size=DATA_SIZE,
        threshold_percentage=THRESHOLD,
        g_ellipsoidal_dimension=ELLIPSOIDAL_DIMENSION,
        alpha=ALPHA,
        random_state=task.random_seed,
        bootstrapping_size=BOOTSTRAPPING_SIZE,
    )
    row: dict[str, object] = {
        "distribution": task.distribution,
        "repetition": task.repetition,
        "random_seed": task.random_seed,
    }
    row.update(
        {constraint_id: float(estimates[index]) for constraint_id, _, index in CONSTRAINTS}
    )
    return row


def _write_atomic(path: Path, rows: list[dict[str, object]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows, columns=RAW_COLUMNS).to_csv(
        temporary, index=False, float_format="%.17g"
    )
    temporary.replace(path)


def generate(output_dir: Path, workers: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    partial_path = output_dir / "raw_results.partial.csv"
    final_path = output_dir / "raw_results.csv"
    tasks = [
        Task(distribution, repetition)
        for distribution in DISTRIBUTIONS
        for repetition in range(REPETITIONS)
    ]
    task_order = {task.key: index for index, task in enumerate(tasks)}
    rows: list[dict[str, object]] = []
    if partial_path.exists():
        rows = pd.read_csv(partial_path).to_dict(orient="records")
        completed = {
            (str(row["distribution"]), int(row["repetition"])) for row in rows
        }
        if len(completed) != len(rows) or not completed.issubset(task_order):
            raise ValueError("Checkpoint has duplicate or unexpected tasks")
        tasks = [task for task in tasks if task.key not in completed]
        LOGGER.info("Resuming from %d rows; %d remain", len(rows), len(tasks))

    started = time.monotonic()
    completed_before = len(rows)
    context = multiprocessing.get_context("spawn")
    with context.Pool(
        processes=workers,
        maxtasksperchild=MAX_TASKS_PER_WORKER,
    ) as pool:
        for completed_now, result in enumerate(
            pool.imap(_run_task, tasks, chunksize=1), start=1
        ):
            rows.append(result)
            completed = completed_before + completed_now
            if completed_now == 1 or completed % 6 == 0 or completed == len(task_order):
                _write_atomic(partial_path, rows)
                elapsed = time.monotonic() - started
                rate = completed_now / max(elapsed, 1e-9)
                eta = (len(task_order) - completed) / rate
                LOGGER.info(
                    "Progress %d/%d (%.1f%%); elapsed %.1fs; ETA %.1fs",
                    completed,
                    len(task_order),
                    100 * completed / len(task_order),
                    elapsed,
                    eta,
                )

    rows.sort(
        key=lambda row: task_order[(str(row["distribution"]), int(row["repetition"]))]
    )
    _write_atomic(final_path, rows)
    partial_path.unlink(missing_ok=True)
    return final_path


def aggregate(raw_path: Path, output_dir: Path) -> Path:
    raw = pd.read_csv(raw_path)
    records: list[dict[str, object]] = []
    for distribution in DISTRIBUTIONS:
        subset = raw.loc[raw["distribution"] == distribution]
        if len(subset) != REPETITIONS:
            raise ValueError(
                f"Expected {REPETITIONS} rows for {distribution}, found {len(subset)}"
            )
        for constraint_id, constraint_latex, _ in CONSTRAINTS:
            values = subset[constraint_id].to_numpy(dtype=float)
            covered = values >= TRUE_VALUE
            estimate_std = float(np.std(values, ddof=1))
            coverage_std = float(np.std(covered.astype(float), ddof=1))
            records.append(
                {
                    "distribution": distribution,
                    "constraint": constraint_id,
                    "constraint_latex": constraint_latex,
                    "repetitions": len(values),
                    "estimate_mean": np.mean(values),
                    "estimate_margin": MONTE_CARLO_Z
                    * estimate_std
                    / np.sqrt(len(values)),
                    "relative_ratio": np.mean(values) / TRUE_VALUE,
                    "relative_ratio_margin": MONTE_CARLO_Z
                    * estimate_std
                    / np.sqrt(len(values))
                    / TRUE_VALUE,
                    "coverage_mean": np.mean(covered),
                    "coverage_margin": MONTE_CARLO_Z
                    * coverage_std
                    / np.sqrt(len(values)),
                }
            )
    summary_path = output_dir / "summary.csv"
    pd.DataFrame.from_records(records).to_csv(
        summary_path, index=False, float_format="%.17g"
    )
    return summary_path


def _scientific(value: float) -> str:
    mantissa, exponent = f"{value:.2E}".split("E")
    return rf"{mantissa}\times 10^{{{int(exponent)}}}"


def render(summary_path: Path, output_dir: Path) -> Path:
    summary = pd.read_csv(summary_path).set_index(["distribution", "constraint"])
    lines: list[str] = []
    latex_lookup = {constraint_id: latex for constraint_id, latex, _ in CONSTRAINTS}
    for distribution in DISTRIBUTIONS:
        for index, constraint_id in enumerate(MANUSCRIPT_ORDER):
            row = summary.loc[(distribution, constraint_id)]
            prefix = (
                rf"\multirow{{6}}{{*}}{{{DISPLAY_NAMES[distribution]}}}"
                if index == 0
                else ""
            )
            lines.append(
                f"{prefix}&${latex_lookup[constraint_id]}$& "
                f"${row.relative_ratio:.2f}\\ (\\pm{row.relative_ratio_margin:.2f})$ & "
                f"${_scientific(row.estimate_mean)}\\ (\\pm {_scientific(row.estimate_margin)})$ & "
                f"${row.coverage_mean:.3f}\\ (\\pm {row.coverage_margin:.3f})$"
                + (
                    r"\\\hline "
                    if index == 5 and distribution != "pareto"
                    else r"\\"
                )
            )
    path = output_dir / "table_f_1a_rows.tex"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _active_subtable(manuscript: str, label: str) -> str:
    pattern = re.compile(
        rf"^[ \t]*\\label\{{{re.escape(label)}\}}[ \t]*$",
        flags=re.MULTILINE,
    )
    matches = list(pattern.finditer(manuscript))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one active {label} label, found {len(matches)}")
    start = manuscript.rfind(r"\begin{subtable}", 0, matches[0].start())
    end = manuscript.index(r"\end{subtable}", matches[0].end())
    return manuscript[start:end]


def _rows(text: str) -> list[str]:
    return [re.sub(r"\s+", "", line) for line in text.splitlines() if r"\pm" in line]


def verify(output_dir: Path, manuscript_path: Path) -> Path:
    generated = _rows((output_dir / "table_f_1a_rows.tex").read_text(encoding="utf-8"))
    expected = _rows(
        _active_subtable(manuscript_path.read_text(encoding="utf-8"), "tb1_tpe")
    )
    if len(generated) != 18 or len(expected) != 18:
        raise RuntimeError(
            f"Expected 18 generated and manuscript rows, got {len(generated)} and {len(expected)}"
        )
    exact = sum(actual == target for actual, target in zip(generated, expected))
    report = [
        "# Table F.1(a) verification report",
        "",
        f"- Manuscript repetitions per distribution: `{REPETITIONS}`",
        "- Existing six-constraint estimator reused without reimplementation",
        "- Bootstrap RNG family: historical Python `random.choices`",
        f"- Exact LaTeX rows after whitespace normalization: `{exact}/18`",
        "",
        "| Row | Status |",
        "| ---: | --- |",
    ]
    failures: list[int] = []
    for index, (actual, target) in enumerate(zip(generated, expected), start=1):
        status = "exact" if actual == target else "different"
        if status == "different":
            failures.append(index)
        report.append(f"| {index} | {status} |")
    if failures:
        report.extend(
            [
                "",
                "A separate numerical comparison is required for rows that are not byte-equivalent after whitespace normalization.",
            ]
        )
    else:
        report.extend(
            [
                "",
                "All aggregate rows match the active manuscript exactly after removing LaTeX whitespace only.",
            ]
        )
    report_path = output_dir / "verification_report.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    if failures:
        raise RuntimeError(
            "Table F.1(a) has non-exact rows requiring numerical review: "
            + ", ".join(map(str, failures))
        )
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("generate", "aggregate", "render", "verify", "all"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manuscript", type=Path, default=DEFAULT_MANUSCRIPT)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    raw_path = args.output_dir / "raw_results.csv"
    summary_path = args.output_dir / "summary.csv"
    if args.stage in {"generate", "all"}:
        raw_path = generate(args.output_dir, args.workers)
    if args.stage in {"aggregate", "all"}:
        summary_path = aggregate(raw_path, args.output_dir)
    if args.stage in {"render", "all"}:
        render(summary_path, args.output_dir)
    if args.stage == "verify":
        verify(args.output_dir, args.manuscript)


if __name__ == "__main__":
    main()
