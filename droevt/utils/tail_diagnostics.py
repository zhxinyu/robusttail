"""
Tail Diagnostics Module
=======================

Motivation
----------
In our distributionally robust tail inference framework, the parameter `a`
represents the onset of the tail region - that is, the point beyond which the
distribution is assumed to satisfy the geometric shape constraints used in the
DRO formulation.  This plays the same conceptual role as the threshold in
classical peaks-over-threshold extreme-value methods.

A recurring practical question raised by practitioners is:
how should this threshold `a` be selected in a principled, data-driven manner?
If `a` is chosen too small, the model may be applied in a region where
tail assumptions do not yet hold; if chosen too large, estimation becomes
inefficient because too few tail observations remain.

Purpose of this module
----------------------
This module provides a set of diagnostic tools to assist analysts in choosing
a reasonable tail threshold in real data applications.  Rather than enforcing
a single fixed rule, the utilities here generate visual diagnostics that allow
the user to assess whether the tail of the empirical distribution exhibits
qualitative features consistent with the modeling assumptions.

In particular, the module implements:

1. The *mean-excess function*
   $e(u) = E[X - u | X > u]$
   whose approximate linearity is a classical indicator of generalized
   Pareto-type tail behavior.

2. A *kernel density estimate* of the empirical tail, together with smoothed
   estimates of its
    - first derivative (monotonicity)
    - second derivative (curvature).
   These diagnostics allow users to visually confirm whether the tail density
   is decreasing and exhibits stable geometric behavior beyond the candidate
   threshold region.

3. A unified plotting routine that overlays candidate threshold values across
   all diagnostics and summarizes the results in a compact figure suitable for
   reporting and reproducibility.

How these tools are intended to be used
---------------------------------------
The user supplies:
    - the observed data `x`
    - a grid of candidate thresholds

The diagnostics help identify the *smallest* threshold beyond which:

    (i)  the mean-excess function becomes approximately linear
    (ii) the estimated density is smoothly decreasing
    (iii) the curvature stabilizes and no longer exhibits strong irregularity.

This mirrors best practice in classical EVT analysis: rather than assuming the
true threshold is known, the analyst evaluates graphical criteria and selects
a value that marks the onset of stable tail behavior.

Role in the paper
-----------------
These diagnostic tools support the empirical sections of the paper and address
reviewer concerns regarding the interpretability and transparency of threshold
selection. They provide a principled and reproducible means of assessing when
our tail shape assumptions become reasonable in practice, while still allowing
for scientific judgment in the final choice of `a`.

This module is not a replacement for statistical modeling, but a complementary
tool for validating tail assumptions prior to applying the proposed DRO
procedure.

"""

from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter1d

############################################################
# Helper: compute mean–excess function   e(u) = E[X − u | X > u]
############################################################
def mean_excess(x: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """
    Compute the mean-excess function $e(u) = E[X - u | X > u]$
    for a given data sample and a list of thresholds.

    Parameters
    ----------
    x : np.ndarray
        1D array of observed data.
    thresholds : np.ndarray
        List or array of thresholds (u) to evaluate the mean-excess at.

    Returns
    -------
    np.ndarray
        Array of mean excess values, one for each threshold $u$ in $thresholds$.
        Returns `np.nan` for any threshold with no exceedances.
    """
    x = np.asarray(x)
    thresholds = np.asarray(thresholds)
    me = []

    for u in thresholds:
        exceedances = x[x > u]
        if exceedances.size == 0:
            me.append(np.nan)
        else:
            me.append(np.mean(exceedances - u))

    return np.array(me)

############################################################
# Helper: compute tail density + smoothed derivatives
############################################################
def tail_density_and_derivatives(
    x: np.ndarray,
    candidate_thresholds: np.ndarray,
    bw_factor: float = 2.0,
    num_grid: int = 400
):
    """
    Estimate the kernel density and its smoothed first and second derivatives
    on a uniform evaluation grid, then interpolate the results at the candidate
    threshold locations.

    Parameters
    ----------
    x : np.ndarray
        1D array of full data sample.
    candidate_thresholds : np.ndarray
        Tail points where density and derivatives should be reported.
    bw_factor : float, optional
        Gaussian smoothing sigma (in grid index units). Typical values 1–3.
    num_grid : int, optional
        Number of grid points for KDE evaluation (default 400).

    Returns
    -------
    grid : np.ndarray
        Uniform evaluation grid.
    dens_tail : np.ndarray
        KDE evaluated at candidate_thresholds.
    d1_tail : np.ndarray
        First derivative evaluated at candidate_thresholds.
    d2_tail : np.ndarray
        Second derivative evaluated at candidate_thresholds.
    """

    x = np.asarray(x)
    candidate_thresholds = np.sort(np.asarray(candidate_thresholds))

    # ---- uniform grid over data range ----
    grid = np.linspace(candidate_thresholds.min() * 0.9, candidate_thresholds.max() * 1.1, num_grid)

    kde = gaussian_kde(x)

    # ---- KDE on grid ----
    density = kde(grid)

    # ---- smooth BEFORE differentiation ----
    density_s = gaussian_filter1d(density, sigma=bw_factor)

    # ---- derivatives wrt x ----
    d1 = np.gradient(density_s, grid)
    d2 = np.gradient(d1, grid)

    # ---- interpolate results at tail thresholds ----
    dens_tail = np.interp(candidate_thresholds, grid, density_s)
    d1_tail = np.interp(candidate_thresholds, grid, d1)
    d2_tail = np.interp(candidate_thresholds, grid, d2)

    return grid, dens_tail, d1_tail, d2_tail

def monotonicity_score(grid, density):
    """
    Fraction of tail grid where the density is decreasing.
    1.0 = perfectly monotone decreasing.
    """
    d = np.diff(density)
    return np.mean(d <= 0)

def curvature_stability_score(second_derivative):
    """
    Stability of curvature sign in the tail.
    Measures fraction of grid where curvature matches the most common sign.
    
    Returns a value between 0 and 1:
    - 1.0 = all non-zero curvatures have the same sign (perfectly stable)
    - 0.5 = equal mix of positive and negative (unstable)
    - Lower values indicate more sign changes (less stable curvature)
    
    Example: If 80% of points have positive curvature and 20% negative,
    returns 0.8 (curvature is mostly stable/positive).
    """
    signs = np.sign(second_derivative)
    signs = signs[signs != 0]  # Remove zeros (flat regions)

    if len(signs) == 0:
        return np.nan

    # Find the most common sign: median of -1/+1 values gives the dominant sign
    # If median is 0, we have equal mix (unstable), so use first sign as reference
    median_sign = np.median(signs)
    modal_sign = int(np.sign(median_sign)) if median_sign != 0 else signs[0]
    
    return np.mean(signs == modal_sign)

def compute_tail_scores(x, candidate_thresholds):
    """
    Compute monotonicity and curvature stability scores for each candidate threshold.
    
    For each threshold, evaluates the tail region (values >= threshold) and computes:
    - monotonicity_score: fraction of tail where density is decreasing
    - curvature_stability_score: fraction of tail where curvature sign is stable
    
    Parameters
    ----------
    x : np.ndarray
        Full data sample (will be sorted internally).
    candidate_thresholds : np.ndarray
        Array of candidate threshold values.
    
    Returns
    -------
    mono_scores : np.ndarray
        Monotonicity scores (0-1) for each threshold.
    conv_scores : np.ndarray
        Curvature stability scores (0-1) for each threshold.
    """
    x = np.sort(np.asarray(x))
    mono_scores = []
    conv_scores = []

    for a in candidate_thresholds:
        # Get tail region for this threshold
        tail_data = x[x >= a]
        if len(tail_data) < 2:
            mono_scores.append(np.nan)
            conv_scores.append(np.nan)
            continue
            
        # Compute density and derivatives on the tail region
        grid, dens, d1, d2 = tail_density_and_derivatives(x, tail_data)
        mono_scores.append(monotonicity_score(grid, dens))
        conv_scores.append(curvature_stability_score(d2))

    return np.array(mono_scores), np.array(conv_scores)

############################################################
# Main plotting utility
############################################################
def plot_tail_diagnostics(
    x: np.ndarray,
    num_thresholds: int = 20,
    tail_fraction: float = 0.45,
    region_name: str = ""
) -> dict[str, Any]:
    """
    Generate a diagnostic plot suite for threshold selection in tail analysis.

    Parameters
    ----------
    x : np.ndarray
        1D array of observed values (data sample).
    tail_fraction : float, optional
        Proportion of largest data used for tail diagnostics 
        (i.e., the upper `tail_fraction` fraction of x), by default 0.3.
    region_name : str, optional
        A label for the dataset or region (used in figure title), by default "".

    Returns
    -------
    dict
        Dictionary containing:
            - 'thresholds': candidate_thresholds,
            - 'mean_excess': mean excess evaluated at thresholds,
            - 'density': smoothed KDE density values,
            - 'density_s': smoothed KDE density values,
            - 'first_derivative': smoothed first derivative of KDE,
            - 'second_derivative': smoothed second derivative of KDE.
    """
    x = np.sort(np.asarray(x))
    n = x.size

    # Tail sample for shape diagnostics (at most 50 points, evenly sampled if more)
    tail_start = int((1 - tail_fraction) * n)
    candidate_thresholds = x[tail_start:]
    if candidate_thresholds.size > num_thresholds:
        # Sample evenly: take indices [0, step, 2*step, ..., -1] where step = size/50
        indices = np.linspace(0, candidate_thresholds.size - 1, 50, dtype=int)
        candidate_thresholds = candidate_thresholds[indices]

    # ====== Compute objects ======
    me_vals = mean_excess(x, candidate_thresholds)
    _, density, first_derivative, second_derivative = tail_density_and_derivatives(x, candidate_thresholds)
    
    # Compute tail scores (monotonicity and curvature stability)
    # mono_scores, conv_scores = compute_tail_scores(x, candidate_thresholds)

    # Convert thresholds to percentiles (percentile of each threshold, i.e., fraction of data <= threshold)
    # Using searchsorted is more efficient than np.sum(x > a) since x is already sorted
    # searchsorted: O(m log n) vs np.sum: O(m*n) for m thresholds
    threshold_percentages = 100 * np.searchsorted(x, candidate_thresholds, side='right') / n

    # ====== Plot ======
    # fig, axs = plt.subplots(3, 2, figsize=(12, 10))
    # ax_me, ax_pdf, ax_d1, ax_d2, ax_scores, ax_conv = axs.flatten()

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    ax_me, ax_pdf, ax_d1, ax_d2 = axs.flatten()

    # --- 1. Mean Excess Plot ---
    ax_me.plot(threshold_percentages, me_vals, lw=2)
    ax_me.set_title("Mean Excess Plot  $e(u)$")
    ax_me.set_xlabel("Threshold Percentile (%)")
    ax_me.set_ylabel("$E[X-u \\,|\\,X>u]$")

    # --- 2. Tail Density ---
    ax_pdf.plot(threshold_percentages, density, lw=2)
    # ax_pdf.plot(candidate_thresholds_percentages, density_s, lw=2, linestyle='--', label="Smoothed KDE")
    ax_pdf.set_title("Tail Density Estimate")
    ax_pdf.set_xlabel("Threshold Percentile (%)")
    ax_pdf.set_ylabel("Density")
    # ax_pdf.legend()

    # --- 3. First Derivative ---
    ax_d1.plot(threshold_percentages, first_derivative, lw=2)
    ax_d1.axhline(0, color="k", ls="--", alpha=0.5)
    ax_d1.set_title("First Derivative  (Monotonicity)")
    ax_d1.set_xlabel("Threshold Percentile (%)")

    # --- 4. Second Derivative ---
    ax_d2.plot(threshold_percentages, second_derivative, lw=2)
    ax_d2.axhline(0, color="k", ls="--", alpha=0.5)
    ax_d2.set_title("Second Derivative  (Curvature)")
    ax_d2.set_xlabel("Threshold Percentile (%)")
    
    # # --- 5. Tail Scores (Monotonicity) ---
    # ax_scores.plot(threshold_percentages, mono_scores, lw=2, label="Monotonicity")
    # ax_scores.set_title("Monotonicity Score")
    # ax_scores.set_xlabel("Threshold Percentile (%)")
    # ax_scores.set_ylabel("Score")
    # ax_scores.set_ylim(0, 1)
    # ax_scores.grid(True, alpha=0.3)
    
    # # --- 6. Curvature Stability Score ---
    # ax_conv.plot(threshold_percentages, conv_scores, lw=2, label="Curvature Stability")
    # ax_conv.set_title("Curvature Stability Score")
    # ax_conv.set_xlabel("Threshold Percentile (%)")
    # ax_conv.set_ylabel("Score")
    # ax_conv.set_ylim(0, 1)
    # ax_conv.grid(True, alpha=0.3)

    if region_name:
        fig.suptitle(f"Tail Diagnostics — {region_name}", fontsize=14)

    fig.tight_layout()

    return {
               "fig": fig,
               "thresholds": candidate_thresholds,
               "threshold_percentages": threshold_percentages,
               "mean_excess": me_vals,
               "density": density,
               "first_derivative": first_derivative,
               "second_derivative": second_derivative,
            #    "monotonicity_scores": mono_scores,
            #    "curvature_stability_scores": conv_scores,
    }