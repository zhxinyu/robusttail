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

1. The *number of exceedances* above each candidate threshold, which helps
   assess whether there are sufficient data points above the threshold for
   reliable estimation and calibration.

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

    (i)  there are sufficient exceedances for reliable estimation
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
from scipy.stats import gaussian_kde, norm, lognorm, gamma, pareto

from .synthetic_data_generator import DISTRIBUTION_DEFAULT_PARAMETERS

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

def gamma_density_derivatives(x, a, scale):
    """
    Compute Gamma density f(x), first derivative f'(x),
    and second derivative f''(x).
    """
    x = np.asarray(x, dtype=float)

    # density
    f = gamma.pdf(x, a=a, scale=scale)

    # first derivative
    f1 = f * ((a - 1.0) / x - 1.0 / scale)

    # second derivative
    f2 = f * (((a - 1.0) / x - 1.0 / scale)**2 - (a - 1.0) / x**2)

    return f, f1, f2

def lognormal_density_derivatives(x, loc, s):
    """
    Lognormal density and derivatives where:

        log X ~ N(loc, s^2)

    Parameters
    ----------
    x : array-like
        Evaluation points (x > 0).
    loc : float
        Mean of log X (mu).
    s : float
        Standard deviation of log X (sigma).

    Returns
    -------
    f : ndarray
        Density f(x).
    f1 : ndarray
        First derivative f'(x).
    f2 : ndarray
        Second derivative f''(x).
    """
    x = np.asarray(x, dtype=float)

    if np.any(x <= 0):
        raise ValueError("Lognormal density is only defined for x > 0.")

    # SciPy mapping: s = sigma, scale = exp(mu), loc = 0
    f = lognorm.pdf(x, s=s, scale=np.exp(loc), loc=0.0)

    A = -1.0 - (np.log(x) - loc) / s**2

    f1 = f * A / x
    f2 = f * (A**2 - A - 1.0 / s**2) / x**2

    return f, f1, f2

def pareto_density_derivatives(x, b, scale=1.0):
    """
    Pareto density and derivatives where:

        f(x) = b * scale^b / x^(b+1),   x >= scale

    This matches scipy.stats.pareto with loc = 0.

    Parameters
    ----------
    x : array-like
        Evaluation points.
    b : float
        Shape (tail index), b > 0.
    scale : float, optional
        Lower bound (x_m), scale > 0.

    Returns
    -------
    f : ndarray
        Density f(x).
    f1 : ndarray
        First derivative f'(x).
    f2 : ndarray
        Second derivative f''(x).
    """
    x = np.asarray(x, dtype=float)

    if scale <= 0:
        raise ValueError("scale must be positive.")
    if b <= 0:
        raise ValueError("b must be positive.")

    # density (SciPy-consistent)
    f = pareto.pdf(x, b=b, scale=scale, loc=0.0)

    # enforce support explicitly
    mask = x >= scale

    f1 = np.zeros_like(f)
    f2 = np.zeros_like(f)

    f1[mask] = - (b + 1.0) * f[mask] / x[mask]
    f2[mask] = (b + 1.0) * (b + 2.0) * f[mask] / x[mask]**2

    return f, f1, f2

############################################################
# Helper: compute tail density + smoothed derivatives
############################################################
def tail_density_and_derivatives(
    x_train: np.ndarray,
    candidate_thresholds: np.ndarray,
    num_grid: int = 200,
    bw_method="scott"
):
    """
    Estimate a univariate probability density and its first and second
    derivatives for positive, right-tailed data using kernel density
    estimation (KDE) in log-space.

    This routine is designed for medium-to-heavy-tailed distributions
    (e.g., Pareto, lognormal, gamma, and other fat-tailed events), where direct
    KDE in the original data space can suffer from boundary bias and severe
    instability in higher-order derivatives.

    The method proceeds as follows:

    1. Log-transform the data:
       Let Y = log(X), where X > 0 is the original random variable.

    2. Fit a Gaussian KDE to Y:
       Denote by g(y) the estimated density of Y.

    3. Compute analytic first and second derivatives of g(y) using the
       closed-form derivatives of the Gaussian kernel.

    4. Transform the density and its derivatives back to the original
       x-space via exact change-of-variables formulas.

    Mathematical formulation
    -------------------------
    Let g(y) be the density of Y = log(X). The density of X is

        f(x) = g(log x) / x ,    x > 0.

    Its first derivative is

        f'(x) = [ g'(log x) - g(log x) ] / x^2 ,

    and its second derivative is

        f''(x) = [ g''(log x) - 3 g'(log x) + 2 g(log x) ] / x^3 .

    These identities are exact consequences of the chain rule and hold
    independently of the choice of kernel or bandwidth.

    Working in log-space greatly improves numerical stability when estimating
    f'(x) and f''(x), particularly in the tail region, and avoids boundary
    artifacts caused by Gaussian kernels leaking mass below the natural
    lower support of X.

    Parameters
    ----------
    x_train : np.ndarray
        One-dimensional array of strictly positive observations drawn from
        the target distribution.

    candidate_thresholds : np.ndarray
        One-dimensional array of positive values at which the density and
        its derivatives are evaluated (e.g., candidate tail thresholds).

    num_grid : int, optional
        Number of equally spaced grid points in log-space used for KDE
        evaluation and differentiation. Default is 400.

    bw_method : str, float, or callable, optional
        Bandwidth specification passed to `scipy.stats.gaussian_kde`.
        Common choices include "scott", "silverman", a scalar multiplier,
        or a callable. Smoother bandwidths are generally required for
        stable estimation of second derivatives.

    Returns
    -------
    x_grid : np.ndarray
        Monotone increasing grid in the original data space on which the
        density and derivatives are evaluated.

    dens_tail : np.ndarray
        Estimated density f(x) evaluated at `candidate_thresholds`.

    d1_tail : np.ndarray
        Estimated first derivative f'(x) evaluated at
        `candidate_thresholds`.

    d2_tail : np.ndarray
        Estimated second derivative f''(x) evaluated at
        `candidate_thresholds`.

    Notes
    -----
    - This function assumes X is strictly positive. Zero or negative values
      must be removed or shifted prior to use.
    - Second-derivative estimates are inherently noisy; meaningful inference
      should rely on persistence (e.g., sustained positivity) or confidence
      bands rather than pointwise sign checks.
    - For Pareto-type tails, convexity of the true density (f''(x) > 0)
      is expected asymptotically; deviations in the estimate reflect finite
      sample variability and smoothing bias.

    References
    ----------
    - Silverman, B. W. (1986). *Density Estimation for Statistics and Data
      Analysis*. Chapman & Hall.
    - Wand, M. P., & Jones, M. C. (1995). *Kernel Smoothing*. Chapman & Hall.
    - Embrechts, P., Klüppelberg, C., & Mikosch, T. (1997). *Modelling
      Extremal Events*. Springer.
    """

    x_train = np.asarray(x_train, dtype=float)
    candidate_thresholds = np.sort(np.asarray(candidate_thresholds, dtype=float))

    # ---- positivity check ----
    if np.any(x_train <= 0):
        raise ValueError("log-space KDE requires strictly positive data.")

    # ---- log-transform ----
    y_train = np.log(x_train)
    y_thresh = np.log(candidate_thresholds)

    # ---- KDE in log-space ----
    kde = gaussian_kde(y_train, bw_method=bw_method)

    # Effective bandwidth in log-space
    h2 = float(kde.covariance.squeeze())
    h = np.sqrt(h2)
    n = y_train.size

    # ---- uniform grid in log-space ----
    x_min = candidate_thresholds.min() * 0.95
    x_max = candidate_thresholds.max() * 1.05
    y_grid = np.linspace(np.log(x_min), np.log(x_max), num_grid)

    # ---- density and derivatives in log-space ----
    def _g(y):
        return kde.evaluate(y)

    def _g1(y):
        y = np.asarray(y)
        z = (y[None, :] - y_train[:, None]) / h
        phi = norm.pdf(z)
        return (1.0 / (n * h**3)) * np.sum((y_train[:, None] - y[None, :]) * phi, axis=0)

    def _g2(y):
        y = np.asarray(y)
        z = (y[None, :] - y_train[:, None]) / h
        phi = norm.pdf(z)
        return (1.0 / (n * h**5)) * np.sum(((y_train[:, None] - y[None, :])**2 - h**2) * phi, axis=0)

    g_val = _g(y_grid)
    g1_val = _g1(y_grid)
    g2_val = _g2(y_grid)

    # ---- transform back to x-space ----
    x_grid = np.exp(y_grid)

    f_val = g_val / x_grid
    f1_val = (g1_val - g_val) / x_grid**2
    f2_val = (g2_val - 3*g1_val + 2*g_val) / x_grid**3

    # ---- interpolate at thresholds ----

    dens_tail = np.interp(y_thresh, y_grid, f_val)
    d1_tail   = np.interp(y_thresh, y_grid, f1_val)
    d2_tail   = np.interp(y_thresh, y_grid, f2_val)

    return x_grid, dens_tail, d1_tail, d2_tail

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
    data_source: tuple[str, dict[str, float]],
    num_thresholds: int = 30,
    tail_fraction: float = 0.45,
    region_name: str = "",
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
            - 'num_exceedances': number of exceedances at each threshold,
            - 'density': smoothed KDE density values,
            - 'first_derivative': smoothed first derivative of KDE,
            - 'second_derivative': smoothed second derivative of KDE.
    """
    x = np.sort(np.asarray(x))
    n = x.size

    # Tail sample for shape diagnostics (at most 50 points, evenly sampled if more)
    tail_start = int((1 - tail_fraction) * n)
    candidate_thresholds = x[tail_start:]
    indices = range(len(candidate_thresholds))

    if candidate_thresholds.size > num_thresholds:
        # Sample evenly: take indices [0, step, 2*step, ..., -1] where step = size/50
        indices = np.linspace(0, candidate_thresholds.size - 1, num_thresholds, dtype=int)
        candidate_thresholds = candidate_thresholds[indices]
    indices = tail_start + np.array(indices)

    # ====== Compute objects ======
    # Compute number of exceedances for each threshold
    num_exceedances = np.array([np.sum(x > threshold) for threshold in candidate_thresholds])
    _, density, first_derivative, second_derivative = tail_density_and_derivatives(x, candidate_thresholds)

    plot_theoretical_density = False
    if data_source[0] in ["gamma", "lognorm", "pareto"]:
        if data_source[0] == "gamma":
            (d_th, d1_th, d2_th) = gamma_density_derivatives(x, **data_source[1])
        elif data_source[0] == "lognorm":
            (d_th, d1_th, d2_th) = lognormal_density_derivatives(x, **data_source[1])
        else:
            (d_th, d1_th, d2_th) = pareto_density_derivatives(x, **data_source[1])

        d_th = d_th[indices]
        d1_th = d1_th[indices]
        d2_th = d2_th[indices]        
        plot_theoretical_density = True

    # Compute tail scores (monotonicity and curvature stability)
    # mono_scores, conv_scores = compute_tail_scores(x, candidate_thresholds)

    # Convert thresholds to percentiles (percentile of each threshold, i.e., fraction of data <= threshold)
    # Using searchsorted is more efficient than np.sum(x > a) since x is already sorted
    # searchsorted: O(m log n) vs np.sum: O(m*n) for m thresholds
    threshold_percentages = 100 * np.searchsorted(x, candidate_thresholds, side='right') / n

    # ====== Plot ======

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    ax_me, ax_pdf, ax_d1, ax_d2 = axs.flatten()

    # --- 1. Number of Exceedances ---
    ax_me.plot(threshold_percentages, num_exceedances, lw=2)
    ax_me.set_title("Number of Exceedances")
    ax_me.set_xlabel("Threshold Percentile (%)")
    ax_me.set_ylabel("Number of Exceedances")

    # --- 2. Tail Density ---
    ax_pdf.plot(threshold_percentages, density, lw=2, label="Kernel Density Estimate")
    ax_pdf.set_title("Tail Density Estimate")
    ax_pdf.set_xlabel("Threshold Percentile (%)")
    # ax_pdf.legend()

    # --- 3. First Derivative ---
    ax_d1.plot(threshold_percentages, first_derivative, lw=2, label="Kernel Density Estimate")
    ax_d1.axhline(0, color="k", ls="--", alpha=0.5)
    ax_d1.set_title("First Derivative  (Monotonicity)")
    ax_d1.set_xlabel("Threshold Percentile (%)")

    # --- 4. Second Derivative ---
    ax_d2.plot(threshold_percentages, second_derivative, lw=2, label="Kernel Density Estimate")
    ax_d2.axhline(0, color="k", ls="--", alpha=0.5)
    ax_d2.set_title("Second Derivative  (Curvature)")
    ax_d2.set_xlabel("Threshold Percentile (%)")
    
    if plot_theoretical_density:
        ax_pdf.plot(threshold_percentages, d_th, lw=2, linestyle='--', color='orange', label="Theoretical") # type: ignore
        ax_pdf.legend(loc='best')
        
        ax_d1.plot(threshold_percentages, d1_th, lw=2, linestyle='--', color='orange', label="Theoretical") # type: ignore
        ax_d1.legend(loc='best')

        ax_d2.plot(threshold_percentages, d2_th, lw=2, linestyle='--', color='orange', label="Theoretical") # type: ignore
        ax_d2.legend(loc='best')

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
               "num_exceedances": num_exceedances,
               "density": density,
               "first_derivative": first_derivative,
               "second_derivative": second_derivative,
            #    "monotonicity_scores": mono_scores,
            #    "curvature_stability_scores": conv_scores,
    }