"""Reproduce the four examples underlying manuscript Figure 1.

The diagnostic calculations and four-panel component plotting are delegated
to ``droevt.utils.tail_diagnostics.plot_tail_diagnostics``.  This module only
provides deterministic data preparation, file output, composite assembly, and
comparison with the stored component figures.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from PIL import Image
from scipy.stats import gamma, lognorm, pareto

from droevt.utils.synthetic_data_generator import (
    DISTRIBUTION_DEFAULT_PARAMETERS,
    generate_synthetic_data,
)
from droevt.utils.tail_diagnostics import plot_tail_diagnostics
from experiments.input_data.cmt.parse_script import parse_ndk

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "experiments" / "generated" / "tail_diagnostics"
STORED_COMPONENT_DIR = REPOSITORY_ROOT / "droevt" / "utils"
MANUSCRIPT_FIGURE = REPOSITORY_ROOT.parent / "latex" / "plots" / "tail_diagnostics.png"

RANDOM_SEED = 20220222
DATA_SIZE = 500
SOURCES = (
    ("gamma", gamma, "Example 1 — Gamma\n(Light Tail)"),
    ("lognorm", lognorm, "Example 2 — Lognormal\n(Sub-Exponential Tail)"),
    ("pareto", pareto, "Example 3 — Pareto\n(Heavy Tail)"),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pixel_comparison(actual: Path, expected: Path) -> tuple[int, int, float]:
    actual_pixels = np.asarray(Image.open(actual).convert("RGBA"), dtype=np.int16)
    expected_pixels = np.asarray(Image.open(expected).convert("RGBA"), dtype=np.int16)
    if actual_pixels.shape != expected_pixels.shape:
        return -1, 255, float("inf")
    difference = np.abs(actual_pixels - expected_pixels)
    return (
        int(np.any(difference != 0, axis=2).sum()),
        int(difference.max()),
        float(difference.mean()),
    )


def _save_diagnostic(
    values: np.ndarray,
    source: tuple[str, dict[str, float]],
    path: Path,
) -> dict[str, np.ndarray]:
    result = plot_tail_diagnostics(values, data_source=source)
    figure = result.pop("fig")
    figure.savefig(path, dpi=100)
    plt.close(figure)
    return {key: np.asarray(value) for key, value in result.items()}


def generate(output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    component_dir = output_dir / "components"
    component_dir.mkdir(parents=True, exist_ok=True)
    arrays_dir = output_dir / "arrays"
    arrays_dir.mkdir(parents=True, exist_ok=True)

    component_paths: list[Path] = []
    diagnostic_arrays: list[dict[str, np.ndarray]] = []
    titles: list[str] = []
    for name, distribution, title in SOURCES:
        parameters = DISTRIBUTION_DEFAULT_PARAMETERS[name]
        values = generate_synthetic_data(
            distribution,
            parameters,
            DATA_SIZE,
            RANDOM_SEED,
        )
        path = component_dir / f"tail_diagnostics_{name}.png"
        arrays = _save_diagnostic(values, (name, parameters), path)
        np.savez(arrays_dir / f"{name}.npz", **arrays)
        component_paths.append(path)
        diagnostic_arrays.append(arrays)
        titles.append(title)

    cmt_values = parse_ndk()["Mw"].to_numpy(dtype=float)
    cmt_path = component_dir / "tail_diagnostics_cmt.png"
    cmt_arrays = _save_diagnostic(cmt_values, ("cmt", {}), cmt_path)
    np.savez(arrays_dir / "cmt.npz", **cmt_arrays)
    component_paths.append(cmt_path)
    diagnostic_arrays.append(cmt_arrays)
    titles.append("Example 4 — Seismic\nMagnitudes (CMT Data)")

    # Render the composite directly rather than shrinking four already-rasterized
    # component figures.  This preserves the original outer dimensions while
    # giving labels and tick marks enough pixels to remain legible in print.
    composite = plt.figure(figsize=(25.26, 6.14), dpi=100)
    grid = composite.add_gridspec(
        2,
        11,
        width_ratios=(1, 1, 0.16, 1, 1, 0.16, 1, 1, 0.16, 1, 1),
        left=0.040,
        right=0.985,
        bottom=0.18,
        top=0.82,
        wspace=0.36,
        hspace=0.08,
    )
    group_columns = ((0, 1), (3, 4), (6, 7), (9, 10))

    for group_index, (arrays, title, columns) in enumerate(
        zip(diagnostic_arrays, titles, group_columns)
    ):
        axes = (
            composite.add_subplot(grid[0, columns[0]]),
            composite.add_subplot(grid[0, columns[1]]),
            composite.add_subplot(grid[1, columns[0]]),
            composite.add_subplot(grid[1, columns[1]]),
        )
        if group_index == 1:
            # Give only the Lognormal right-hand panels a little extra room for
            # their longer y-tick labels; keep every other pair compact.
            for axis in (axes[1], axes[3]):
                position = axis.get_position()
                axis.set_position(
                    [
                        position.x0 + 0.003,
                        position.y0,
                        position.width - 0.003,
                        position.height,
                    ]
                )
        threshold = arrays["threshold_percentages"]
        series = (
            arrays["num_exceedances"],
            arrays["density"],
            arrays["first_derivative"],
            arrays["second_derivative"],
        )
        theoretical = (
            np.array([]),
            arrays["theoretical_density"],
            arrays["theoretical_first_derivative"],
            arrays["theoretical_second_derivative"],
        )

        for panel_index, (axis, values, theory) in enumerate(
            zip(axes, series, theoretical)
        ):
            axis.plot(threshold, values, lw=2.2, color="tab:blue")
            if theory.size:
                axis.plot(
                    threshold,
                    theory,
                    lw=2.2,
                    linestyle="--",
                    color="tab:orange",
                )
            if panel_index in (2, 3):
                axis.axhline(0, color="0.35", lw=1.2, linestyle="--", alpha=0.7)
            axis.tick_params(axis="both", labelsize=17, pad=2)
            axis.xaxis.set_major_locator(MaxNLocator(nbins=3))
            axis.yaxis.set_major_locator(MaxNLocator(nbins=3))
            axis.grid(axis="y", color="0.92", linewidth=0.8)
            for spine in axis.spines.values():
                spine.set_color("0.65")
                spine.set_linewidth(0.9)

        for axis in axes[:2]:
            axis.tick_params(axis="x", labelbottom=False)

        left = group_index / 4 + 0.012
        composite.text(
            left,
            0.96,
            title,
            ha="left",
            va="top",
            fontsize=25,
            fontweight="bold",
            linespacing=1.05,
        )

    for x_position in (0.25, 0.50, 0.75):
        composite.add_artist(
            plt.Line2D(
                (x_position, x_position),
                (0.07, 0.97),
                transform=composite.transFigure,
                color="0.90",
                lw=1.0,
            )
        )

    composite.legend(
        handles=(
            Line2D((0,), (0,), color="tab:blue", lw=2.2),
            Line2D((0,), (0,), color="tab:orange", lw=2.2, linestyle="--"),
        ),
        labels=(
            "Empirical exceedances / kernel estimate",
            "Theoretical density / derivatives",
        ),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=2,
        frameon=False,
        fontsize=22,
        handlelength=2.6,
        columnspacing=1.6,
    )
    composite_path = output_dir / "tail_diagnostics.png"
    composite.savefig(composite_path, dpi=100)
    plt.close(composite)
    return component_dir, composite_path


def verify(output_dir: Path) -> Path:
    report = [
        "# Figure 1 verification report",
        "",
        "- Scientific generator: `droevt.utils.tail_diagnostics.plot_tail_diagnostics`",
        f"- Synthetic data size: `{DATA_SIZE}`",
        f"- Synthetic random seed: `{RANDOM_SEED}`",
        "- CMT input: all `Mw` values returned by the existing `parse_ndk()` parser",
        "",
        "| Component | Byte comparison | Differing pixels | Max channel difference | Mean absolute channel difference |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for name in ("gamma", "lognorm", "pareto", "cmt"):
        filename = f"tail_diagnostics_{name}.png"
        generated = output_dir / "components" / filename
        expected = STORED_COMPONENT_DIR / filename
        exact = _sha256(generated) == _sha256(expected)
        differing, maximum, mean = _pixel_comparison(generated, expected)
        report.append(
            f"| `{name}` | {'exact' if exact else 'different'} | {differing} | "
            f"{maximum} | {mean:.8g} |"
        )

    generated_composite = output_dir / "tail_diagnostics.png"
    report.extend(
        [
            "",
            f"- Generated composite dimensions: `{Image.open(generated_composite).size}`",
            f"- Manuscript composite dimensions: `{Image.open(MANUSCRIPT_FIGURE).size}`",
            "- The historical composite assembler was not retained; composite byte equality is not claimed.",
            "- Acceptance is based on deterministic regeneration of all four scientific components and substantive visual equivalence of the four-column composite.",
            f"- Regeneration Matplotlib version: `{matplotlib.__version__}`",
        ]
    )
    report_path = output_dir / "verification_report.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("generate", "verify", "all"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    if args.stage in {"generate", "all"}:
        generate(args.output_dir)
    if args.stage == "verify":
        verify(args.output_dir)


if __name__ == "__main__":
    main()
