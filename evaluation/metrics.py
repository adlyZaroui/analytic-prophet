"""Metrics shared by more than one tier.

The Hessian-weighted distance is the reason this module exists rather than
living inside Tier 1: the metrics table in #79 lists it under Tiers 1 *and* 2,
and one implementation is better than two that drift.
"""
import numpy as np

from analytic_prophet.optimizer import finite_difference_hessian


def smooth_hessian(model, theta, design=None):
    """The curvature of the posterior at `theta`, without the Laplace term.

    The L1 prior is piecewise linear, so its second derivative is zero
    everywhere it is defined and undefined on the kink -- differencing across
    that kink would report 1/(tau*h), about 1e9, and swamp everything else. It
    is dropped here and accounted for linearly by `quadratic_change` below,
    which is the same split `newton()` and `projected_newton()` make.

    Central differences of the *analytic* gradient, so this inherits the
    gradient's accuracy rather than an objective's.
    """
    design = design if design is not None else model._design_matrices()
    gradient = lambda point: model._gradient(point, include_l1_prior=False,
                                             design=design)
    return finite_difference_hessian(gradient, np.asarray(theta, dtype=float))


def quadratic_change(model, theta, other, design=None):
    """The second-order prediction of how much worse `other` is than `theta`.

        f(other) - f(theta)  ~=  g.d  +  d'Hd/2  +  (the L1 term, exactly)

    with `d = other - theta`. This is the quantity that makes a parameter-space
    distance mean something: raw distance is dominated by the directions the
    data does not identify -- `k` against `delta`, which trade off almost
    freely -- so two fits can be far apart and be the same function. Weighted
    by the curvature, the distance *is* the loss gap, to second order.

    Returns the predicted change in the minus-log-posterior; positive means
    `other` is the worse point.
    """
    theta = np.asarray(theta, dtype=float)
    other = np.asarray(other, dtype=float)
    design = design if design is not None else model._design_matrices()
    step = other - theta

    gradient = model._gradient(theta, include_l1_prior=False, design=design)
    hessian = smooth_hessian(model, theta, design=design)
    smooth = float(gradient @ step + 0.5 * step @ hessian @ step)

    # the Laplace term is linear in |delta|, so it is exact rather than expanded
    delta = model.layout.delta
    l1 = float(np.sum(np.abs(other[delta])) - np.sum(np.abs(theta[delta]))) \
        / model.changepoint_prior_scale
    return smooth + l1


def quadratic_distance(model, reference, point, design=None):
    """Curvature-weighted distance from `reference` to `point`.

        d'Hd / 2,   H evaluated at `reference`,  d = point - reference

    The distinction from `quadratic_change` matters and is easy to lose. That
    one is a *loss gap*: expanded around a fitted point it carries the gradient
    term and therefore rewards whichever optimizer reached the lower objective.
    Comparing two implementations' "distance from the truth" that way is a
    tautology -- the one with the better posterior is further from the truth by
    construction, whatever it actually recovered.

    This is a *distance*: one ruler, fixed at `reference`, applied to both. Use
    it with the truth as the reference when asking who recovered it.
    """
    reference = np.asarray(reference, dtype=float)
    step = np.asarray(point, dtype=float) - reference
    hessian = smooth_hessian(model, reference, design=design)
    return float(0.5 * step @ hessian @ step)


def curvature_spectrum(model, theta, design=None):
    """Eigenvalues of the posterior curvature, largest first.

    The near-zero end is the flat directions -- the thing this project has
    described qualitatively since the beginning and never put a number on.

    Measured on 300 days of the Peyton Manning series, linear growth spans
    8.1e5 down to 5.4e-2, a condition number of 1.5e7. Under **flat growth the
    trend block is exactly singular**: its smallest eigenvalue is 0, because the
    likelihood never sees `k` or `delta` there and the Laplace prior that holds
    `delta` contributes no curvature at all. That is the README's "flat growth
    removes those directions" as a number rather than a story -- and it is why
    flat growth ties with Prophet exactly while linear growth does not.

    Expect a few eigenvalues at -1e-11 or so on a singular block: they are zeros
    plus differencing noise, not negative curvature. A caller wanting a
    condition number should clip at zero and say so.
    """
    hessian = smooth_hessian(model, theta, design=design)
    return np.sort(np.linalg.eigvalsh(0.5 * (hessian + hessian.T)))[::-1]


def paired_summary(differences):
    """Median, IQR and a signed-rank p-value for paired per-series differences.

    A mean across series is the wrong summary here: forecast errors are
    heavy-tailed and one series dominates it. Paired, because both
    implementations see the same series and the same splits, so the pairing
    removes the series-to-series variation that would otherwise drown the
    effect being measured.
    """
    values = np.asarray([d for d in differences if np.isfinite(d)], dtype=float)
    summary = {
        "n": int(values.size),
        "median": float(np.median(values)) if values.size else float("nan"),
        "iqr_low": float(np.percentile(values, 25)) if values.size else float("nan"),
        "iqr_high": float(np.percentile(values, 75)) if values.size else float("nan"),
        "wins": int(np.sum(values < 0)),
        "losses": int(np.sum(values > 0)),
        "p_value": float("nan"),
    }
    if values.size >= 6 and np.any(values != 0):
        from scipy.stats import wilcoxon
        summary["p_value"] = float(wilcoxon(values).pvalue)
    return summary
