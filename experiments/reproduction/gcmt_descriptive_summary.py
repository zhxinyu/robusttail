"""Reproduce the GCMT descriptive statistics and box plots in Table G.2.

This is a thin presentation wrapper around the curated GCMT inputs already
shipped in ``experiments/input_data/cmt/parsed_data``.  Run from the repository
root with the ``rs`` environment:

    python -m experiments.reproduction.gcmt_descriptive_summary all
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = REPOSITORY_ROOT / "experiments" / "input_data" / "cmt" / "parsed_data"
HISTORICAL_PLOT_DIR = (
    REPOSITORY_ROOT / "experiments" / "input_data" / "cmt" / "plots"
)
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "experiments" / "generated" / "gcmt_descriptive_summary"
DEFAULT_MANUSCRIPT = REPOSITORY_ROOT.parent / "latex" / "jasa_manu.tex"
MANUSCRIPT_PLOT_DIR = REPOSITORY_ROOT.parent / "latex" / "plots"

REGIONS: tuple[tuple[str, str, str], ...] = (
    ("ECUADOR", "Ecuador", "ecuador_boxplot.png"),
    (
        "OFF_COAST_OF_NORTHERN_CA",
        "Off Coast of Northern CA, US",
        "off_coast_of_northern_ca_boxplot.png",
    ),
    ("TURKEY", "Turkey", "turkey_boxplot.png"),
    (
        "HOKKAIDO_JAPAN_REGION",
        "Hokkaido, Japan Region",
        "hokkaido_japan_region_boxplot.png",
    ),
    ("BANDA_SEA", "Banda Sea", "banda_sea_boxplot.png"),
    ("KURIL_ISLANDS", "Kuril Islands", "kuril_islands_boxplot.png"),
    ("SOLOMON_ISLANDS", "Solomon Islands", "solomon_islands_boxplot.png"),
    (
        "FIJI_ISLANDS_REGION",
        "Fiji Islands Regions",
        "fiji_islands_region_boxplot.png",
    ),
)


def _load_region(region: str) -> np.ndarray:
    values = np.loadtxt(INPUT_DIR / f"{region}.csv", dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"Expected a nonempty one-dimensional sample for {region}")
    return values


def generate_summary(output_dir: Path) -> Path:
    """Compute the exact descriptive quantities printed in Table G.2."""
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for region, display_name, plot_filename in REGIONS:
        values = _load_region(region)
        q95, q99, q995 = np.quantile(values, [0.95, 0.99, 0.995])
        records.append(
            {
                "region": region,
                "display_name": display_name,
                "q95": q95,
                "q99": q99,
                "q995": q995,
                "maximum": np.max(values),
                "events": values.size,
                "plot_filename": plot_filename,
            }
        )
    path = output_dir / "summary.csv"
    pd.DataFrame.from_records(records).to_csv(path, index=False, float_format="%.17g")
    return path


def render_table(summary_path: Path, output_dir: Path) -> Path:
    """Render the data rows embedded in the manuscript table."""
    summary = pd.read_csv(summary_path)

    def values(column: str) -> str:
        return " & ".join(f"{value:.2f}" for value in summary[column])

    lines = [
        rf"95\% Quantile & {values('q95')} \\",
        rf"99\% Quantile & {values('q99')} \\",
        rf"99.5\% Quantile & {values('q995')} \\",
        rf"Max & {values('maximum')} \\",
        "Events count & "
        + " & ".join(str(int(value)) for value in summary["events"])
        + r"\\",
    ]
    path = output_dir / "table_rows.tex"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def render_plots(output_dir: Path) -> Path:
    """Regenerate the eight compact horizontal box plots.

    The visual settings recover the stored plot design.  The manuscript assets
    identify Matplotlib 3.7.1 in their PNG metadata; byte equality is therefore
    checked but not assumed under a different Matplotlib build.
    """
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for region, _, plot_filename in REGIONS:
        values = _load_region(region)
        figure, axis = plt.subplots(figsize=(7.4, 1.8), dpi=100)
        axis.boxplot(
            values,
            vert=False,
            widths=0.75,
            patch_artist=True,
            boxprops={"facecolor": "orangered"},
            medianprops={"color": "yellow"},
            flierprops={
                "marker": "o",
                "markerfacecolor": "black",
                "markeredgecolor": "black",
            },
        )
        axis.set_xticks([])
        axis.set_yticks([])
        # Recover the 10-pixel frame used by the 740 x 180 manuscript assets.
        # Stating it explicitly avoids version-dependent tight-layout changes.
        figure.subplots_adjust(
            left=10 / 740,
            right=730 / 740,
            bottom=10 / 180,
            top=170 / 180,
        )
        figure.savefig(plot_dir / plot_filename, dpi=100)
        plt.close(figure)
    return plot_dir


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pixel_comparison(actual: Path, expected: Path) -> tuple[int, int, float]:
    actual_pixels = np.asarray(Image.open(actual).convert("RGBA"), dtype=np.int16)
    expected_pixels = np.asarray(Image.open(expected).convert("RGBA"), dtype=np.int16)
    if actual_pixels.shape != expected_pixels.shape:
        return -1, 255, float("inf")
    absolute = np.abs(actual_pixels - expected_pixels)
    differing = int(np.any(absolute != 0, axis=2).sum())
    return differing, int(absolute.max()), float(absolute.mean())


def _active_table(manuscript: str, label: str) -> str:
    match = re.search(
        rf"^[ \t]*\\label\{{{re.escape(label)}\}}[ \t]*$",
        manuscript,
        flags=re.MULTILINE,
    )
    if match is None:
        raise ValueError(f"Active manuscript label not found: {label}")
    start = manuscript.rfind(r"\begin{table}", 0, match.start())
    end = manuscript.index(r"\end{table}", match.end())
    return manuscript[start:end]


def _normalized_data_rows(text: str) -> list[str]:
    prefixes = ("95\\% Quantile", "99\\% Quantile", "99.5\\% Quantile", "Max", "Events count")
    rows: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(prefixes):
            rows.append(re.sub(r"\s+", "", stripped))
    return rows


def verify(output_dir: Path, manuscript_path: Path) -> Path:
    manuscript = manuscript_path.read_text(encoding="utf-8")
    expected_rows = _normalized_data_rows(_active_table(manuscript, "tb5_real_tb0"))
    generated_rows = _normalized_data_rows(
        (output_dir / "table_rows.tex").read_text(encoding="utf-8")
    )
    table_exact = generated_rows == expected_rows

    report = [
        "# Table G.2 verification report",
        "",
        f"- Numerical LaTeX rows byte-normalized exact: `{table_exact}`",
        "- Input: curated `experiments/input_data/cmt/parsed_data/*.csv` files",
        "",
        "| Plot | Stored experiment asset vs manuscript | Regenerated vs manuscript | Differing pixels | Max channel difference | Mean absolute channel difference |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    regenerated_all_exact = True
    stored_all_exact = True
    for _, _, plot_filename in REGIONS:
        stored = HISTORICAL_PLOT_DIR / plot_filename
        regenerated = output_dir / "plots" / plot_filename
        expected = MANUSCRIPT_PLOT_DIR / plot_filename
        stored_exact = _sha256(stored) == _sha256(expected)
        regenerated_exact = _sha256(regenerated) == _sha256(expected)
        differing, maximum, mean = _pixel_comparison(regenerated, expected)
        stored_all_exact &= stored_exact
        regenerated_all_exact &= regenerated_exact
        report.append(
            f"| `{plot_filename}` | {'exact' if stored_exact else 'different'} | "
            f"{'exact' if regenerated_exact else 'different'} | {differing} | "
            f"{maximum} | {mean:.8g} |"
        )

    report.extend(
        [
            "",
            f"- All eight stored experiment assets are byte-identical to the manuscript assets: `{stored_all_exact}`",
            f"- All eight regenerated assets are byte-identical to the manuscript assets: `{regenerated_all_exact}`",
            f"- Regeneration Matplotlib version: `{plt.matplotlib.__version__}`",
            "- Stored manuscript PNG metadata reports Matplotlib `3.7.1`.",
        ]
    )
    report_path = output_dir / "verification_report.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    if not table_exact:
        raise AssertionError("Generated Table G.2 numerical rows differ from manuscript")
    if not stored_all_exact:
        raise AssertionError("Stored experiment box plots differ from manuscript assets")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("generate", "render", "verify", "all"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manuscript", type=Path, default=DEFAULT_MANUSCRIPT)
    args = parser.parse_args()

    if args.stage in {"generate", "all"}:
        generate_summary(args.output_dir)
        render_plots(args.output_dir)
    if args.stage in {"render", "all"}:
        render_table(args.output_dir / "summary.csv", args.output_dir)
    if args.stage == "verify":
        verify(args.output_dir, args.manuscript)


if __name__ == "__main__":
    main()
