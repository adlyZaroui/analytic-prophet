"""Tier 1 -- parameter recovery on synthetic data (#79).

**Who recovers the true model better?** On real data that question has no
answer: there is no true parameter vector, so agreement between two fits is all
anyone can measure. Generated from the model, there is one.

The tier reports two kinds of distance from the truth, and the contrast between
them is the point.

  * the **raw** error per parameter block, which is what a reader expects;
  * the **identified** error -- `metrics.quadratic_change`, the same distance
    weighted by the curvature of the posterior, which to second order *is* the
    loss gap.

They disagree, and the disagreement is the finding rather than a caveat. `k` and
`delta` trade off almost freely, so a fit can be far from the truth in `delta`
and describe the same function; raw distance reports that as failure, weighted
distance reports it as the flat direction it is.

Both implementations fit the same generated series, and the generator places its
changepoints where both will place theirs (see synthetic.py), so `delta` indexes
the same breakpoints on all three vectors.

**The identified error is in nats and does not pool across series.** The
curvature it weights by grows with the sample, so the same parameter error costs
more on a longer series. It compares two implementations *on one series*; a mean
over series of different lengths measures how much data each one had. The report
pairs per series for exactly this reason.
"""
import tempfile

import numpy as np

import harness
import metrics
import synthetic
from harness import Measurement

SIZES = (100, 300, 1000)
NOISES = (0.05, 0.2)
SEEDS = (1, 2, 3)


def _blocks(layout):
    return {
        "k": np.array([layout.k_idx]),
        "m": np.array([layout.m_idx]),
        "delta": np.arange(layout.delta.start, layout.delta.stop),
        "sigma_obs": np.array([layout.sigma_obs_idx]),
        "beta": np.arange(layout.beta.start, layout.beta.stop),
    }


def _prophet_theta(prophet_model, layout):
    """Prophet's fit in our canonical layout.

    Directly comparable without any projection: since #15 the changepoints are
    identical and since #36 the Fourier basis is, so the two vectors index the
    same quantities in the same order.
    """
    params = prophet_model.params
    return np.concatenate((
        np.ravel(params["k"])[:1],
        np.ravel(params["m"])[:1],
        np.ravel(params["delta"]),
        np.ravel(params["sigma_obs"])[:1],
        np.ravel(params["beta"]),
    ))


def _one(truth, lib_path):
    from prophet import Prophet

    from analytic_prophet import AnalyticProphet

    ours = AnalyticProphet()
    ours.fit_cpp(truth.frame, lib_path=lib_path)
    layout = ours.layout
    theta_ours = ours.get_parameters().copy()

    prophet_model = Prophet(**harness.PROPHET_KWARGS)
    prophet_model.fit(truth.frame)
    theta_prophet = _prophet_theta(prophet_model, layout)

    design = ours._design_matrices()
    objective = lambda theta: ours._minus_log_posterior(theta, design=design)
    rows = []

    def row(implementation, metric, value, unit=""):
        rows.append(Measurement(1, truth.label, "linear", implementation,
                                metric, float(value), unit))

    # the scales the two implementations normalized by must agree, or every
    # comparison below is partly a comparison of divisors
    row("both", "y_scale_abs_diff", abs(ours.y_scale - prophet_model.y_scale))
    row("both", "changepoints", layout.n_changepoints)
    row("both", "design_columns", layout.n_regressor_columns)
    # the generating vector's own sparsity, for reference rather than as a
    # target: MAP with an L1 prior over nested, collinear step functions is not
    # a support-recovery procedure, so neither fit is expected to match it (#88)
    row("both", "sum_abs_delta_true", np.abs(truth.theta[truth.layout.delta]).sum())
    row("both", "exact_zeros_true", truth.layout.n_changepoints - truth.active)

    if theta_prophet.size != theta_ours.size:
        row("both", "layout_mismatch", 1.0)
        return rows
    row("both", "layout_mismatch", 0.0)

    for name, theta in (("analytic_prophet", theta_ours), ("prophet", theta_prophet)):
        error = theta - truth.theta
        for block, index in _blocks(layout).items():
            row(name, f"{block}_error_l2", np.linalg.norm(error[index]))
        row(name, "theta_error_l2", np.linalg.norm(error))
        # Two threshold-free readouts of the trend's sparsity, replacing a
        # count of `|delta| > 1e-6` that was almost entirely a function of that
        # threshold on Prophet's side (#95). `sum_abs_delta` is what the Laplace
        # prior actually charges for; `exact_zeros` is what distinguishes the
        # two optimizers, and needs no cutoff to say so.
        rates = np.abs(theta[layout.delta])
        row(name, "sum_abs_delta", rates.sum())
        row(name, "exact_zeros", np.sum(rates == 0.0))
        row(name, "minus_log_posterior", objective(theta), "nats")
        # How much better than the truth this point scores on the sample it was
        # fitted to. Negative for any MAP estimate worth the name, and *more*
        # negative is not better -- it is the fit exploiting its own sample
        # harder, which is what overfitting looks like from inside.
        row(name, "excess_over_truth", objective(theta) - objective(truth.theta), "nats")
        # The recovery metric: a curvature-weighted distance from the truth,
        # with the curvature taken *at the truth* so both implementations are
        # measured with the same ruler. Weighted rather than raw because raw
        # distance is dominated by the directions the data does not identify --
        # k against delta -- where being far away costs nothing and means
        # nothing.
        row(name, "identified_error", metrics.quadratic_distance(
            ours, truth.theta, theta, design=design), "nats")

    spectrum = metrics.curvature_spectrum(ours, theta_ours, design=design)
    positive = spectrum[spectrum > 0]
    row("both", "curvature_max", spectrum[0])
    row("both", "curvature_min_positive", positive[-1] if positive.size else float("nan"))
    row("both", "curvature_condition", spectrum[0] / positive[-1] if positive.size else float("nan"))
    return rows


def collect(sizes=SIZES, noises=NOISES, seeds=SEEDS, lib_path=None):
    lib_path = lib_path or harness.build_extension(tempfile.mkdtemp())
    measurements = []
    for size in sizes:
        for noise in noises:
            for seed in seeds:
                truth = synthetic.generate(size, noise=noise,
                                           seed=harness.SEED + seed)
                measurements += _one(truth, lib_path)
    return measurements


harness.register(1, "recovery", collect)
