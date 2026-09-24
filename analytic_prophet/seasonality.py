"""The Fourier basis, the seasonality registry, and Prophet's selection rule.

[fc] these live on the Prophet class itself; they are free functions here
because they depend on nothing but their arguments, which is also what makes
them testable without a fitted model.
"""
import logging

import numpy as np
import pandas as pd

from .constants import SIGMA

logger = logging.getLogger("analytic_prophet")


# An entry mirrors Prophet's own, field for field -- [fc] add_seasonality:
#     {period, fourier_order, prior_scale, mode, condition_name}
# Every field is honored as of #16 task 11. check_seasonality_supported still
# validates them, since a value outside the allowed set would otherwise reach
# the design matrix rather than the caller.
SEASONALITY_DEFAULTS = {"prior_scale": SIGMA, "mode": "additive", "condition_name": None}

# Of those, the ones the fit actually reads. The rest are checked rather than
# ignored -- see check_seasonality_supported.
HONORED_SEASONALITY_FIELDS = ("period", "fourier_order", "prior_scale",
                              "condition_name", "mode")

def seasonality(period, fourier_order, **overrides):
    """One registry entry, with Prophet's defaults filled in."""
    return {"period": float(period), "fourier_order": int(fourier_order),
            **SEASONALITY_DEFAULTS, **overrides}

BUILT_IN_NAMES = ("daily", "weekly", "yearly")

# Column names a component may not take, because predict() and the Stan data
# already use them. [fc] validate_column_name, which also derives the _lower
# and _upper variants -- those are uncertainty-interval columns in the output
# frame, so a component named after one would collide there rather than here.
_RESERVED_STEMS = ("trend", "additive_terms", "daily", "weekly", "yearly",
                   "holidays", "zeros", "extra_regressors_additive", "yhat",
                   "extra_regressors_multiplicative", "multiplicative_terms")

RESERVED_COLUMN_NAMES = frozenset(
    _RESERVED_STEMS
    + tuple(stem + "_lower" for stem in _RESERVED_STEMS)
    + tuple(stem + "_upper" for stem in _RESERVED_STEMS)
    + ("ds", "y", "cap", "floor", "y_scaled", "cap_scaled"))

# Prophet's built-in seasonalities, as period and default Fourier order.
# [fc] set_auto_seasonalities. Which of them a model fits is decided from the
# history -- see AUTO_SEASONALITY_RULES.
BUILT_IN_SEASONALITIES = {
    "yearly": seasonality(365.25, 10),
    "weekly": seasonality(7.0, 3),
    "daily": seasonality(1.0, 4),
}

def history_spacing(ds):
    """(first, last, min_dt) -- what the auto-selection rule decides from.

    [fc] set_auto_seasonalities computes these on `self.history`, which is
    sorted by ds, so the spacing is sorted here too rather than relying on the
    caller's row order. Zero spacings (duplicate timestamps) are excluded, and
    the leading NaT of `diff()` drops out of `min()`, both as Prophet does.
    """
    ds = pd.to_datetime(pd.Series(np.asarray(ds)).sort_values())
    dt = ds.diff()
    return ds.min(), ds.max(), dt.iloc[dt.values.nonzero()[0]].min()

def parse_seasonality_args(name, arg, auto_disable, default_order, seasonalities):
    """Fourier order for a built-in seasonality, or 0 to leave it out.

    [fc] Prophet.parse_seasonality_args, branch for branch. `arg is True` and
    `arg is False` are identity checks in the original, so `weekly=1` asks for
    order 1 rather than the default 3 -- matched here deliberately.
    """
    if arg == 'auto':
        fourier_order = 0
        if name in seasonalities:
            logger.info("Found custom seasonality named %r, disabling built-in "
                        "%r seasonality.", name, name)
        elif auto_disable:
            logger.info("Disabling %s seasonality. Run with %s_seasonality=True "
                        "to override this.", name, name)
        else:
            fourier_order = default_order
    elif arg is True:
        fourier_order = default_order
    elif arg is False:
        fourier_order = 0
    else:
        fourier_order = int(arg)
    return fourier_order

# The auto-selection rule, one row per built-in. [fc] set_auto_seasonalities:
# yearly needs two years of history; weekly and daily additionally need the
# data to be spaced more finely than the period they describe, since a weekly
# component cannot be identified from weekly-or-coarser observations.
#
# `disable` is read as "leave this one out", so it is the negation of what the
# docstring there states in the positive.
AUTO_SEASONALITY_RULES = (
    ("yearly", "yearly_seasonality", 365.25, 10, pd.Timedelta(days=730), None),
    ("weekly", "weekly_seasonality", 7.0, 3, pd.Timedelta(weeks=2), pd.Timedelta(weeks=1)),
    ("daily", "daily_seasonality", 1.0, 4, pd.Timedelta(days=2), pd.Timedelta(days=1)),
)

# [fc] set_auto_seasonalities, verbatim. This regime is not hypothetical: with
# yearly forced on a 328-day slice, this implementation and Prophet diverged
# 111% over a 30-day forecast while our posterior was the better one. Following
# the rule above is what closed that gap -- the warning is for a user who
# overrides it anyway.
UNDER_IDENTIFIED_WARNING = (
    "Yearly seasonality is enabled with less than 730 days (approximately 2 "
    "years) of history. The model may be under-identified, and the "
    "trend/seasonality decomposition can be unstable and dependent on the "
    "Prophet/Stan version. Consider disabling yearly seasonality or providing "
    "more history.")

def check_seasonality_supported(seasonalities):
    """Reject registry fields the fit does not yet honor.

    Nothing is rejected outright any more -- `prior_scale` became honored in
    task 5, `condition_name` in task 7 and `mode` in task 11 -- so this is now
    a validity check rather than a refusal. It still earns its place: an
    unrecognized mode would otherwise reach the design matrix as "not
    multiplicative", i.e. silently additive.
    """
    for name, props in seasonalities.items():
        for field, default in SEASONALITY_DEFAULTS.items():
            value = props.get(field, default)
            if field in HONORED_SEASONALITY_FIELDS:
                continue
            if value != default:
                raise NotImplementedError(
                    f"seasonality {name!r} sets {field}={value!r}, which this "
                    f"implementation does not honor yet (only {field}={default!r} "
                    f"is fitted). Tracked in #16.")
        scale = float(props.get("prior_scale", SIGMA))
        if not scale > 0:
            raise ValueError(
                f"seasonality {name!r} has prior_scale={scale!r}; the prior "
                f"normal(0, prior_scale) needs a positive scale")
        mode = props.get("mode", "additive")
        if mode not in ("additive", "multiplicative"):
            raise ValueError(
                f"seasonality {name!r} has mode={mode!r}; it must be "
                f'"additive" or "multiplicative"')

def condition_masks(seasonalities, df):
    """Validate each component's condition column and return it as a bool array.

    [fc] setup_dataframe. Returns name -> mask for the conditioned components
    only; a component without a `condition_name` is absent from the result and
    applies everywhere.

    `isin([True, False])` is Prophet's own test, and it accepts 0/1 as well as
    True/False because `1 == True` in pandas -- so an integer indicator column
    works, while NaN does not.
    """
    masks = {}
    for name, props in seasonalities.items():
        condition_name = props.get("condition_name")
        if condition_name is None:
            continue
        if condition_name not in df:
            raise ValueError(f"Condition {condition_name!r} missing from dataframe")
        column = df[condition_name]
        if not column.isin([True, False]).all():
            raise ValueError(f"Found non-boolean in column {condition_name!r}")
        masks[name] = column.astype(bool).to_numpy()
    return masks

def make_seasonality_features(dates, period, series_order, prefix):
    """Fourier features for one seasonal component, as a named frame.

    [fc] Prophet.make_seasonality_features, including the `{prefix}_delim_{i}`
    column names -- they carry no meaning for the fit, which sees an array, but
    they are what makes a design matrix here readable beside one from there.
    """
    features = fourier_series(seasonal_time(dates), period, series_order)
    columns = [f"{prefix}_delim_{i + 1}" for i in range(features.shape[1])]
    return pd.DataFrame(features, columns=columns)

def condition_matrix(seasonalities, masks, n_rows):
    """The masks as a T x n_components matrix, in registry order, for the C++.

    An empty matrix means no component is conditioned, which is the common case
    and the one the C++ takes as its default. Unconditioned components are all
    ones, so the C++ applies one uniform rule rather than carrying an index of
    which components have a condition.
    """
    if not masks:
        return np.empty((n_rows, 0))
    return np.column_stack([
        np.asarray(masks[name], dtype=float) if name in masks else np.ones(n_rows)
        for name in seasonalities
    ])

def seasonality_columns(seasonalities):
    """K -- the total number of seasonality columns the registry implies."""
    return sum(2 * props["fourier_order"] for props in seasonalities.values())

def seasonality_modes(seasonalities):
    """1 where a column is multiplicative, 0 where additive, per column.

    [stan] `vector[K] s_a` / `vector[K] s_m`; [fc] make_all_seasonality_features
    records a mode per component and regressor_column_matrix spreads it over
    that component's columns. Every column of a component shares its mode, so
    this repeats each one across its block.
    """
    return np.concatenate([
        np.full(2 * props["fourier_order"],
                1.0 if props.get("mode", "additive") == "multiplicative" else 0.0)
        for props in seasonalities.values()
    ]) if seasonalities else np.empty(0)

def seasonality_prior_scales(seasonalities):
    """`sigmas` -- one prior scale per column of the design matrix.

    [stan] `vector[K] sigmas` in the data block, `beta ~ normal(0, sigmas)` in
    the model block; [fc] make_all_seasonality_features extends prior_scales by
    `[props['prior_scale']] * features.shape[1]`. Every column of a component
    shares that component's scale, so this repeats each one across its block.
    """
    return np.concatenate([
        np.full(2 * props["fourier_order"], float(props["prior_scale"]))
        for props in seasonalities.values()
    ]) if seasonalities else np.empty(0)

# Prophet measures seasonal time in days since the Unix epoch, not since the
# start of the series. [fc] Prophet.fourier_series
FOURIER_EPOCH = pd.Timestamp("1970-01-01")

YEARLY_PERIOD = 365.25

def seasonal_time(ds):
    """Days since the 1970 epoch, the clock Prophet builds Fourier features on.

    Using the series start instead -- as this did before -- shifts the phase of
    every frequency. The fitted seasonality is unchanged (an orthogonal
    rotation per frequency, and `beta`'s prior is rotation-invariant), but the
    coefficients are not comparable with Prophet's, which is what this fixes.
    """
    return (pd.to_datetime(ds) - FOURIER_EPOCH).dt.total_seconds().to_numpy() / (24 * 60 * 60)

def fourier_series(t_days, period, n):
    """Fourier features, column-for-column identical to Prophet's.

    Ordering matters as much as the clock: Prophet interleaves, putting
    sin(order i) at column 2i and cos(order i) at 2i+1. This emitted all
    cosines then all sines, which permutes `beta` even when the phase agrees.
    """
    t_days = np.asarray(t_days, dtype=float)

    # `2*pi/period` is folded into one constant. Prophet instead computes
    # `2*pi*t` and scales by `(i+1)/period`. The two are the same function in
    # exact arithmetic and differ by up to 7e-12 in floating point, because `t`
    # is days since 1970 and the angles reach ~15000 radians. Keeping this form
    # is a deliberate choice -- see README, "Where this deviates on purpose".
    angles = (2 * np.pi / period) * np.outer(t_days, np.arange(1, n + 1))
    result = np.empty((t_days.shape[0], 2 * n))
    result[:, 0::2] = np.sin(angles)
    result[:, 1::2] = np.cos(angles)
    return result
