import glob
import importlib
import importlib.util
import logging
import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import halfcauchy
from typing import Tuple

logger = logging.getLogger("customProphet")

# ---------------------------------------------------------------------------
# Model constants, checked against facebook/prophet (additive mode, linear
# growth, MAP). Sources:
#   [stan] python/stan/prophet.stan
#   [fc]   python/prophet/forecaster.py
#
# Three categories, kept deliberately distinct:
#
#   FITTED  free parameters in Stan's `parameters` block, estimated by L-BFGS.
#           Here: k, m, delta(S), sigma_obs, beta(K) -- see the layout below.
#   SCALE   the *scale of a prior* on a fitted parameter. A scale is not the
#           value of anything; it controls how hard a fitted parameter is
#           pulled toward zero. Prophet estimates none of these -- there are no
#           hierarchical priors in the model.
#   FIXED   structural constants (counts, ranges, orders), passed to Stan as
#           data.
# ---------------------------------------------------------------------------

# FIXED -- number of changepoints S. [fc] Prophet.__init__ n_changepoints=25
N_CHANGE_POINTS = 25
# FIXED -- changepoints span the first 80% of history.
# [fc] Prophet.__init__ changepoint_range=0.8
CHANGEPOINT_RANGE = 0.8
# FIXED -- Fourier order N for yearly seasonality, giving 2N = 20 columns.
# [fc] set_auto_seasonalities, yearly fourier_order=10
# K is no longer this: a model's seasonality registry decides it (#16 task 2),
# and this only sets what the default registry puts in the yearly slot.
n_yearly = 10

# SCALE on delta, the changepoint rate adjustments: delta ~ double_exponential(0, tau).
# [stan] model block; [fc] Prophet.__init__ changepoint_prior_scale=0.05 (user-configurable)
TAU = 0.05
# SCALE on beta, the seasonality coefficients: beta ~ normal(0, sigmas).
# [stan] model block; [fc] Prophet.__init__ seasonality_prior_scale=10.0 (user-configurable)
# This is the model-wide default a seasonality inherits when it carries no
# prior_scale of its own. The vector Stan actually receives is built per column
# by seasonality_prior_scales() -- `sigmas` is vector[K] there, so seasonality,
# holiday and regressor terms can each have their own.
SIGMA = 10.0
# SCALE on sigma_obs, the observation noise: sigma_obs ~ normal(0, 0.5),
# truncated at 0 by `real<lower=0>`. [stan] parameters + model blocks.
# Hardcoded in prophet.stan -- not user-configurable.
SIGMA_OBS_PRIOR_SCALE = 0.5
# SCALE on k: k ~ normal(0, 5). [stan] model block. Hardcoded, not configurable.
sigma_k = 5
# SCALE on m: m ~ normal(0, 5). [stan] model block. Hardcoded, not configurable.
sigma_m = 5

# Stan's L-BFGS convergence criteria, with the CmdStan defaults Prophet runs
# under: CmdStanPyBackend.fit calls optimize(algorithm='LBFGS', iter=int(1e4))
# and sets no tolerances or history_size.
# [stan] src/stan/optimization/bfgs.hpp, ConvergenceOptions + step()
#
# Stan stops as soon as ANY of five tests holds: absolute and relative
# objective change, absolute and relative gradient norm, and parameter change.
# L-BFGS-B exposes the first and third of those; the C++ core evaluates four
# of the five directly (see optimize.cpp). Matching them is what stops both
# paths running tens of thousands of iterations past convergence.
STAN_EPS = 2.220446049250313e-16   # machine epsilon, as Stan uses it
STAN_TOL_REL_OBJ = 1e+4            # tol_rel_obj, scaled by STAN_EPS
STAN_TOL_GRAD = 1e-8               # tol_grad
STAN_MAX_ITERATIONS = 10000        # Prophet passes iter=int(1e4)

# INIT -- Prophet overrides Stan's random init with explicit values, so Stan's
# `init_r * N(0, 1)` default is never reached. [fc] calculate_initial_params
# returns sigma_obs=1.0, delta=zeros(S), beta=zeros(K), and k/m from
# linear_growth_init (see below).
SIGMA_OBS_INIT = 1.0

class ParameterLayout:
    """Where each block sits in the parameter vector, given S changepoints and
    K seasonality columns: [k, m, delta(S), sigma_obs, beta(K)].

    This used to be five module constants fixed at import, which pinned the
    model to 25 changepoints and one order-10 seasonality. The C++ core stopped
    assuming either in #3; carrying the layout as a value makes the Python side
    follow, which is what lets a seasonality registry exist at all.
    """

    __slots__ = ("n_changepoints", "n_seasonality_columns", "n_holiday_columns",
                 "k_idx", "m_idx", "delta", "sigma_obs_idx", "beta", "size")

    def __init__(self, n_changepoints, n_seasonality_columns, n_holiday_columns=0):
        self.n_changepoints = n_changepoints
        self.n_seasonality_columns = n_seasonality_columns
        self.n_holiday_columns = n_holiday_columns
        n_seasonality_columns += n_holiday_columns
        self.k_idx = 0
        self.m_idx = 1
        self.delta = slice(2, 2 + n_changepoints)
        self.sigma_obs_idx = 2 + n_changepoints
        self.beta = slice(3 + n_changepoints, 3 + n_changepoints + n_seasonality_columns)
        self.size = 3 + n_changepoints + n_seasonality_columns

    @property
    def n_regressor_columns(self):
        """K -- every column of the design matrix, seasonal and holiday alike.

        The objective does not distinguish them: `beta` spans the lot and the
        prior is per column either way. The split is kept only so the two
        blocks can be built and sliced separately.
        """
        return self.n_seasonality_columns + self.n_holiday_columns

    @property
    def holidays(self):
        """The holiday coefficients' slice of the *parameter vector*.

        Indexed like `beta` and `delta`, so it applies to `opt_params`. For the
        design matrix and `sigmas` -- which are indexed by column, from 0 -- use
        `holiday_block`. The two differ by the 3 + S offset, and mixing them
        silently returns the wrong slice rather than raising.
        """
        return slice(self.beta.start + self.n_seasonality_columns, self.beta.stop)

    @property
    def holiday_block(self):
        """The holiday columns' slice of the *design matrix* and of `sigmas`."""
        return slice(self.n_seasonality_columns, self.n_regressor_columns)

    @property
    def seasonality_block(self):
        """The seasonal columns' slice of the design matrix and of `sigmas`."""
        return slice(0, self.n_seasonality_columns)

    def __repr__(self):
        return (f"ParameterLayout(S={self.n_changepoints}, "
                f"K={self.n_regressor_columns} "
                f"({self.n_seasonality_columns} seasonal + "
                f"{self.n_holiday_columns} holiday), size={self.size})")


# The layout a default model uses: 25 changepoints, yearly seasonality at
# order 10. The module constants below are its fields, kept so that callers
# reading a default-shaped vector need no layout in hand.
DEFAULT_LAYOUT = ParameterLayout(N_CHANGE_POINTS, 2 * n_yearly)
K_IDX = DEFAULT_LAYOUT.k_idx
M_IDX = DEFAULT_LAYOUT.m_idx
DELTA_SLICE = DEFAULT_LAYOUT.delta
SIGMA_OBS_IDX = DEFAULT_LAYOUT.sigma_obs_idx
BETA_SLICE = DEFAULT_LAYOUT.beta

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


def seasonality_design_matrix(t_seasonality, seasonalities, masks=None):
    """Fourier features for every registered component, concatenated.

    Column order follows the registry's insertion order, and within a component
    Prophet's own interleaving. With one yearly component this is exactly the
    matrix built before the registry existed.

    A component named in `masks` has the rows where its mask is False zeroed --
    [fc] `features[~df[props['condition_name']]] = 0`. The columns stay in the
    matrix, so `beta` keeps its width and only the rows the condition excludes
    stop contributing.
    """
    masks = masks or {}
    blocks = []
    for name, props in seasonalities.items():
        block = fourier_series(t_seasonality, props["period"], props["fourier_order"])
        if name in masks:
            block = np.where(np.asarray(masks[name], dtype=bool)[:, None], block, 0.0)
        blocks.append(block)
    return np.concatenate(blocks, axis=1) if blocks else np.empty((len(t_seasonality), 0))


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


# [fc] make_holidays.get_country_holidays_class. The one substitution Prophet
# carries, for callers still passing Turkey as 'TU'.
COUNTRY_CODE_SUBSTITUTIONS = {"TU": "TR"}

# [fc] get_holiday_names sweeps these years to enumerate a country's names.
COUNTRY_NAME_SWEEP = range(1995, 2045)

HOLIDAYS_PACKAGE_HINT = (
    "country holidays need the `holidays` package, which is not installed. "
    "`pip install holidays`, or pass a holidays frame to add_holidays() "
    "instead.")


def _country_holidays_class(country):
    """The `holidays` class for a country code, imported lazily.

    Lazily because the rest of the model does not need the package: a user
    fitting trend and seasonality should not be required to install it, and a
    missing import should surface here rather than at import time.
    """
    try:
        import holidays as holidays_package
    except ImportError as exc:                    # pragma: no cover
        raise ImportError(HOLIDAYS_PACKAGE_HINT) from exc

    country = COUNTRY_CODE_SUBSTITUTIONS.get(country, country)
    if not hasattr(holidays_package, country):
        raise AttributeError(f"Holidays in {country} are not currently supported!")
    return getattr(holidays_package, country)


def get_holiday_names(country):
    """Every holiday name the country can produce. [fc] get_holiday_names.

    Swept over a fixed window of years rather than the data's, because the
    names are validated once when the country is registered -- before any
    frame has been seen.
    """
    return set(_country_holidays_class(country)(
        language="en_US", years=np.arange(COUNTRY_NAME_SWEEP.start,
                                          COUNTRY_NAME_SWEEP.stop)).values())


def make_holidays_df(years, country):
    """A holidays frame for `years`. [fc] make_holidays_df.

    `expand=False` keeps the observed-date variants from being generated as
    separate entries, and one date can carry several names, so the list column
    is exploded into one row each.
    """
    generated = _country_holidays_class(country)(
        expand=False, language="en_US", years=list(years))
    frame = pd.DataFrame([(date, generated.get_list(date)) for date in generated],
                         columns=["ds", "holiday"])
    frame = frame.explode("holiday").reset_index(drop=True)
    frame["ds"] = pd.to_datetime(frame["ds"])
    return frame


def regressor_standardization(column, standardize):
    """(mu, std) for one regressor. [fc] initialize_scales.

    'auto' standardizes unless the column is binary, since centring a 0/1
    indicator turns "the flag is on" into two values neither of which is zero,
    and its coefficient stops meaning what it did. A constant column is left
    alone too -- there is nothing to divide by.
    """
    values = pd.to_numeric(column)
    if len(values.unique()) < 2:
        standardize = False
    elif standardize == "auto":
        standardize = set(values.unique()) != {1, 0}

    if not standardize:
        return 0.0, 1.0
    return float(values.mean()), float(values.std())


def regressor_columns(extra_regressors, df):
    """The standardized regressor columns, in registration order.

    [fc] setup_dataframe validates and `df[name] = (df[name] - mu) / std`
    applies the standardization fitted on the history -- at predict time too,
    with the *fitted* mu and std rather than the future frame's own.
    """
    if not extra_regressors:
        return np.empty((len(df), 0))

    columns = []
    for name, props in extra_regressors.items():
        if name not in df:
            raise ValueError(f"Regressor {name!r} missing from dataframe")
        values = pd.to_numeric(df[name])
        if values.isnull().any():
            raise ValueError(f"Found NaN in column {name!r}")
        columns.append((values.to_numpy(dtype=float) - props["mu"]) / props["std"])
    return np.column_stack(columns)


def validate_holidays_frame(holidays, validate_name):
    """Check and normalize the holidays frame. [fc] Prophet.validate_inputs.

    `validate_name` is the model's validate_column_name, passed in so the
    reserved-name and collision checks are the same ones add_seasonality uses.
    """
    if holidays is None:
        return None
    if not (isinstance(holidays, pd.DataFrame) and "ds" in holidays
            and "holiday" in holidays):
        raise ValueError('holidays must be a DataFrame with "ds" and "holiday" columns.')

    holidays = holidays.copy()
    holidays["ds"] = pd.to_datetime(holidays["ds"])
    if holidays["ds"].isnull().any() or holidays["holiday"].isnull().any():
        raise ValueError("Found a NaN in holidays dataframe.")

    has_lower = "lower_window" in holidays
    has_upper = "upper_window" in holidays
    if has_lower + has_upper == 1:
        raise ValueError("Holidays must have both lower_window and upper_window, or neither")
    if has_lower:
        if holidays["lower_window"].max() > 0:
            raise ValueError("Holiday lower_window should be <= 0")
        if holidays["upper_window"].min() < 0:
            raise ValueError("Holiday upper_window should be >= 0")

    for name in holidays["holiday"].unique():
        # check_holidays=False: a holiday may not collide with a seasonality or
        # regressor, but the frame is allowed to name the same holiday twice --
        # that is how a recurring holiday lists its occurrences.
        validate_name(name, check_holidays=False)
    return holidays


def make_holiday_features(dates, holidays, default_prior_scale):
    """Indicator columns for every holiday occurrence and window offset.

    [fc] Prophet.make_holiday_features. Returns (features, prior_scales, names):
    a T x H float array, one prior scale per column, and the holiday names in
    the order they were first seen.

    One column per (holiday, offset) pair, named `holiday_delim_+n` or
    `holiday_delim_-n`, and the columns are **sorted by name** -- Prophet sorts
    them, so `beta` is indexed by that order rather than by the frame's row
    order, and anything else would misalign the coefficients.

    A window offset that falls outside `dates` still gets its column, all
    zeros: the fit and the forecast must present the same columns even when a
    holiday happens not to land in one of them.
    """
    dates = pd.to_datetime(pd.Series(np.asarray(dates)))
    if holidays is None or len(holidays) == 0:
        return np.empty((len(dates), 0)), [], []

    # a holiday's date matched to the day, not the timestamp
    row_index = pd.DatetimeIndex(dates.dt.date)

    columns = {}
    prior_scales = {}
    for row in holidays.itertuples():
        if pd.isnull(row.ds):
            # a training holiday with no occurrence in this frame: its columns
            # are created below by the offsets of its other rows, or stay absent
            continue
        try:
            lower = int(getattr(row, "lower_window", 0))
            upper = int(getattr(row, "upper_window", 0))
        except ValueError:
            lower = upper = 0

        scale = float(getattr(row, "prior_scale", default_prior_scale))
        if np.isnan(scale):
            scale = float(default_prior_scale)
        if row.holiday in prior_scales and prior_scales[row.holiday] != scale:
            raise ValueError(
                f"Holiday {row.holiday!r} does not have consistent prior scale "
                f"specification.")
        if scale <= 0:
            raise ValueError("Prior scale must be > 0")
        prior_scales[row.holiday] = scale

        for offset in range(lower, upper + 1):
            key = f"{row.holiday}_delim_{'+' if offset >= 0 else '-'}{abs(offset)}"
            column = columns.setdefault(key, np.zeros(len(dates)))
            occurrence = pd.to_datetime(row.ds.date() + pd.Timedelta(days=offset))
            matches = np.flatnonzero(row_index == occurrence)
            column[matches] = 1.0

    if not columns:
        return np.empty((len(dates), 0)), [], list(prior_scales)

    names = sorted(columns)
    features = np.column_stack([columns[name] for name in names])
    scales = [prior_scales[name.split("_delim_")[0]] for name in names]
    return features, scales, list(prior_scales)


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

# [stan] `int trend_indicator` in the data block: 0 linear, 1 logistic, 2 flat.
TREND_INDICATORS = {"linear": 0, "logistic": 1, "flat": 2}


def flat_growth_init(y_scaled):
    """[fc] flat_growth_init: no rate, and the offset at the mean of y."""
    return 0.0, float(np.mean(y_scaled))


def logistic_gamma_and_jacobian(k, m, delta, t_change):
    """Piecewise offsets for the logistic trend, and their Jacobian.

    [stan] logistic_gamma. Each segment's offset is chosen so the curve stays
    continuous where the rate changes:

        k_s        = [k, k + cumsum(delta)]                 S+1 segment rates
        gamma[i]   = (t_change[i] - m_pr_i) * (1 - k_s[i]/k_s[i+1])
        m_pr_{i+1} = m_pr_i + gamma[i],   m_pr_1 = m

    Unlike the linear case, where gamma is just `-t_change * delta`, this is a
    recursion: gamma[i] depends on every earlier gamma through `m_pr`. So its
    Jacobian is accumulated forward alongside it rather than written down.

    Returns (gamma, dgamma) with dgamma of shape (S, 2 + S), columns ordered
    (k, m, delta).
    """
    delta = np.asarray(delta, dtype=float)
    n = len(delta)
    k_s = np.concatenate(([k], k + np.cumsum(delta)))

    # d k_s[i] / d(k, m, delta): 1 for k, 0 for m, and delta_j enters k_s[i]
    # for every j < i
    dk_s = np.zeros((n + 1, 2 + n))
    dk_s[:, 0] = 1.0
    for i in range(1, n + 1):
        dk_s[i, 2:2 + i] = 1.0

    gamma = np.empty(n)
    dgamma = np.zeros((n, 2 + n))
    m_pr = m
    dm_pr = np.zeros(2 + n)
    dm_pr[1] = 1.0

    for i in range(n):
        c = 1.0 - k_s[i] / k_s[i + 1]
        dc = -(dk_s[i] * k_s[i + 1] - k_s[i] * dk_s[i + 1]) / k_s[i + 1] ** 2

        gamma[i] = (t_change[i] - m_pr) * c
        dgamma[i] = -dm_pr * c + (t_change[i] - m_pr) * dc

        m_pr = m_pr + gamma[i]
        dm_pr = dm_pr + dgamma[i]
    return gamma, dgamma


def logistic_trend_and_jacobian(k, m, delta, t_scaled, cap_scaled, A, t_change):
    """The logistic trend and d(trend)/d(k, m, delta).

    [stan] logistic_trend: cap .* inv_logit((k + A*delta) .* (t - (m + A*gamma))).

    Returns (trend, jacobian) with jacobian of shape (T, 2 + S). The caller
    contracts it with the residual; keeping the full Jacobian here rather than
    the contracted gradient is what lets the multiplicative multiplier be
    applied outside, exactly as it is for the linear trend.
    """
    gamma, dgamma = logistic_gamma_and_jacobian(k, m, delta, t_change)
    rate = k + np.dot(A, delta)
    offset = m + np.dot(A, gamma)
    z = rate * (t_scaled - offset)

    # exp(-|z|) form: the naive 1/(1+exp(-z)) overflows for z very negative
    sigmoid = np.where(z >= 0, 1.0 / (1.0 + np.exp(-np.abs(z))),
                       np.exp(-np.abs(z)) / (1.0 + np.exp(-np.abs(z))))
    trend = cap_scaled * sigmoid

    d_offset = np.dot(A, dgamma)
    d_offset[:, 1] += 1.0                      # offset = m + A*gamma
    d_rate = np.zeros((len(t_scaled), 2 + len(delta)))
    d_rate[:, 0] = 1.0
    d_rate[:, 2:] = A

    dz = d_rate * (t_scaled - offset)[:, None] - rate[:, None] * d_offset
    jacobian = (cap_scaled * sigmoid * (1.0 - sigmoid))[:, None] * dz
    return trend, jacobian


def logistic_growth_init(t_scaled, y_scaled, cap_scaled):
    """[fc] logistic_growth_init: put the curve through the first and last
    points, clamping y into (0, cap) first so the logs are defined."""
    i0, i1 = int(np.argmin(t_scaled)), int(np.argmax(t_scaled))
    span = t_scaled[i1] - t_scaled[i0]

    c0, c1 = cap_scaled[i0], cap_scaled[i1]
    y0 = max(0.01 * c0, min(0.99 * c0, y_scaled[i0]))
    y1 = max(0.01 * c1, min(0.99 * c1, y_scaled[i1]))

    r0, r1 = c0 / y0, c1 / y1
    if abs(r0 - r1) <= 0.01:
        r0 = 1.05 * r0

    l0, l1 = np.log(r0 - 1), np.log(r1 - 1)
    return (l0 - l1) / span, l0 * span / (l0 - l1)


def linear_growth_init(t_scaled, y_scaled):
    """Prophet's deterministic starting point for (k, m): the line through the
    first and last points of the scaled series.

    [fc] Prophet.linear_growth_init -- it indexes by argmin/argmax of ds rather
    than assuming the frame is sorted, so this does the same via t_scaled.

        k = (y_scaled[i1] - y_scaled[i0]) / (t[i1] - t[i0])
        m = y_scaled[i0] - k * t[i0]

    Prophet always passes this in explicitly, which is why Stan's random
    `init_r * N(0, 1)` default never applies.
    """
    i0 = int(np.argmin(t_scaled))
    i1 = int(np.argmax(t_scaled))
    span = t_scaled[i1] - t_scaled[i0]
    k = (y_scaled[i1] - y_scaled[i0]) / span
    m = y_scaled[i0] - k * t_scaled[i0]
    return float(k), float(m)

def det_dot(a, b):
    return (a * b[None, :]).sum(axis=-1)

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

def extract_params(params, layout=DEFAULT_LAYOUT):
    k = params[layout.k_idx]
    m = params[layout.m_idx]
    delta = params[layout.delta]
    sigma_obs = params[layout.sigma_obs_idx]
    beta = params[layout.beta]
    return k, m, delta, sigma_obs, beta

def from_dict_to_array(params, layout=DEFAULT_LAYOUT):
    """Pack a parameter dict into a vector in `layout`'s order.

    `beta` used to be overwritten with zeros here regardless of what was
    passed (#34), which was invisible only because every caller happened to
    pass zeros.
    """
    beta = np.asarray(params['beta'], dtype=float)
    if beta.shape != (layout.n_regressor_columns,):
        raise ValueError(
            f"beta has {beta.shape} entries but the layout expects "
            f"{layout.n_regressor_columns}")
    return np.concatenate(([params['k']], [params['m']], np.asarray(params['delta'], dtype=float),
                           [params['sigma_obs']], beta))

CPP_MODULE_NAME = 'analytic_prophet_cpp'

BUILD_HINT = (
    "Build it from legacy/optimize.cpp -- see the compile command in that file's "
    "trailing comment, or let tests/conftest.py's compiled_optimizer_module "
    "fixture build it for you."
)

_cpp_module_cache = {}

def load_cpp_module(lib_path=None):
    """Import the compiled pybind11 extension backing fit_cpp().

    With lib_path=None this is an ordinary import, so a built or installed
    extension is found on sys.path like any other module; failing that, it
    looks for one built in place next to this file. Passing lib_path loads a
    specific .so, which is how the tests point at one built into a temp dir.

    The ctypes binding this replaced hardcoded a *relative* path
    ('./liboptimization.so'), so it only worked when the process happened to be
    running from the right directory.
    """
    if lib_path is None:
        try:
            return importlib.import_module(CPP_MODULE_NAME)
        except ImportError:
            here = os.path.dirname(os.path.abspath(__file__))
            candidates = sorted(glob.glob(os.path.join(here, CPP_MODULE_NAME + '*.so')))
            if not candidates:
                raise ImportError(
                    f"{CPP_MODULE_NAME} is not importable and no build of it was found "
                    f"in {here}. {BUILD_HINT}"
                )
            lib_path = candidates[0]

    lib_path = os.path.abspath(lib_path)
    if lib_path not in _cpp_module_cache:
        spec = importlib.util.spec_from_file_location(CPP_MODULE_NAME, lib_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"{lib_path} is not loadable as a Python extension module. {BUILD_HINT}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _cpp_module_cache[lib_path] = module
    return _cpp_module_cache[lib_path]

def canonical_to_cpp(params, layout=DEFAULT_LAYOUT):
    """(k, m, delta, sigma_obs, beta) -> (k, m, delta, beta, zeta).

    The C++ core carries zeta = log(sigma_obs) as the LAST element, whereas the
    canonical layout keeps sigma_obs between delta and beta, mirroring the order
    of Stan's `parameters` block. Both describe the same model; only the
    packing differs. The log is what keeps sigma_obs positive in the C++, since
    liblbfgs has no box constraints.
    """
    k, m, delta, sigma_obs, beta = extract_params(params, layout)
    return np.concatenate(([k], [m], delta, beta, [np.log(sigma_obs)]))

def cpp_to_canonical(params, layout=DEFAULT_LAYOUT):
    """Inverse of canonical_to_cpp: sigma_obs = exp(zeta), moved into place."""
    params = np.asarray(params, dtype=float)
    n_delta = layout.n_changepoints
    k, m = params[0], params[1]
    delta = params[2:2 + n_delta]
    beta = params[2 + n_delta:-1]
    sigma_obs = np.exp(params[-1])
    return np.concatenate(([k], [m], delta, [sigma_obs], beta))

def canonical_to_split(params, layout=DEFAULT_LAYOUT):
    """(k, m, delta, sigma_obs, beta) -> (k, m, delta_pos, delta_neg, sigma_obs, beta).

    The Laplace prior on delta puts |delta|/tau in the objective, which is not
    differentiable at delta=0 -- and that is exactly where the optimum sits,
    since the prior is what drives most changepoint rates to zero. L-BFGS-B
    assumes a smooth objective and stalls on those kinks well short of the
    optimum (while still reporting success).

    Splitting delta into non-negative parts, delta = delta_pos - delta_neg,
    turns |delta| into delta_pos + delta_neg: smooth, with the non-smoothness
    pushed into simple bound constraints L-BFGS-B handles natively. At an
    optimum at most one of each pair is non-zero, so the two problems have the
    same solution. This is the same fix the C++ core gets from OWL-QN.
    """
    k, m, delta, sigma_obs, beta = extract_params(params, layout)
    return np.concatenate(([k], [m], np.maximum(delta, 0), np.maximum(-delta, 0), [sigma_obs], beta))

def split_to_canonical(z, n_delta=N_CHANGE_POINTS):
    """Inverse of canonical_to_split: delta = delta_pos - delta_neg."""
    k, m = z[0], z[1]
    delta_pos = z[2:2 + n_delta]
    delta_neg = z[2 + n_delta:2 + 2 * n_delta]
    sigma_obs = z[2 + 2 * n_delta]
    beta = z[3 + 2 * n_delta:]
    return np.concatenate(([k], [m], delta_pos - delta_neg, [sigma_obs], beta))

def predict_trend(k, m, delta, t_change, t_scaled, y_scale,
                  cap_scaled=None, floor=None, growth='linear'):
    """The trend in normalized-y space, de-normalized once at the end.

    Shared by predict() and trend_forecast_uncertainty() so the
    de-normalization can't drift apart between the two again. `cap_scaled`
    selects logistic growth; without it the trend is piecewise linear.
    """
    A = (t_scaled[:, None] >= t_change) * 1
    if growth == 'flat':
        trend_normalized = np.full(len(t_scaled), m)
    elif cap_scaled is None:
        gamma = -t_change * delta
        trend_normalized = (k + det_dot(A, delta)) * t_scaled + (m + det_dot(A, gamma))
    else:
        trend_normalized = logistic_trend_and_jacobian(
            k, m, delta, t_scaled, cap_scaled, A, t_change)[0]
    trend = trend_normalized * y_scale
    return trend if floor is None else trend + floor

class CustomProphet:

    def __init__(self, growth='linear', changepoints=None, n_changepoints=N_CHANGE_POINTS,
                 changepoint_range=CHANGEPOINT_RANGE, yearly_seasonality='auto',
                 weekly_seasonality='auto', daily_seasonality='auto', holidays=None,
                 seasonality_mode='additive', seasonality_prior_scale=SIGMA,
                 holidays_prior_scale=SIGMA, changepoint_prior_scale=TAU,
                 mcmc_samples=0, interval_width=0.8, uncertainty_samples=1000,
                 stan_backend=None, scaling='absmax', holidays_mode=None):
        """Argument for argument, [fc] Prophet.__init__.

        Arguments for things this implementation does not do are accepted and
        then *rejected*, rather than being absent: a script ported from Prophet
        should fail where it is actually wrong, not at the first AttributeError
        for something that would have been ignored anyway.

        `changepoint_prior_scale` is Prophet's name for what the Stan model and
        every derivation in this repository call `tau`. The argument takes
        Prophet's name and the attribute keeps Stan's.
        """
        self.rng = np.random.default_rng()

        self.t_scaled = None
        self.y = None
        self.y_scaled = None
        self.y_scale = None
        self.ds = None

        self.T = None
        self.growth = growth
        self.cap_scaled = None
        self.floor = None
        self.n_changepoints = n_changepoints
        self.t_change = None
        self.changepoint_range = changepoint_range

        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.seasonalities = {}
        self._auto_registered = set()

        self.extra_regressors = {}
        self.holidays = None
        self.holidays_prior_scale = holidays_prior_scale
        self.holidays_mode = holidays_mode
        self.train_holiday_names = None
        self.country_holidays = None
        self._fitted_with_cpp = False
        self._fit_lib_path = None
        self._regressor_name = None
        self._regressor_history = None
        self._holiday_columns = 0
        self._holiday_prior_scales = []
        self._data_columns = None
        self._data_column_count = 0
        self._data_prior_scales = []
        self._data_modes = []
        self.condition_masks = {}

        self.seasonality_mode = seasonality_mode
        self.seasonality_prior_scale = seasonality_prior_scale
        self.layout = DEFAULT_LAYOUT

        # Prophet's name on the argument, Stan's on the attribute.
        self.tau = changepoint_prior_scale
        self.sigmas = seasonality_prior_scales({})
        self.s_m = np.empty(0)
        self.s_a = np.empty(0)
        self._multiplicative = False
        self.sigma_obs = SIGMA_OBS_INIT

        self.interval_width = interval_width
        self.uncertainty_samples = uncertainty_samples

        self.m = None
        self.k = None
        self.delta = None
        self.beta = None

        self.opt_params = None
        self.loss_over_iterations = None
        self.opt = None

        self.sigma_k = sigma_k
        self.sigma_m = sigma_m

        self.t_seasonality = None

        self._reject_unsupported(changepoints, mcmc_samples, stan_backend, scaling)
        if holidays is not None:
            self.add_holidays(holidays)

    @staticmethod
    def _reject_unsupported(changepoints, mcmc_samples, stan_backend, scaling):
        """Refuse what Prophet accepts and this does not.

        Accepting these silently would be the failure mode the whole of #16
        exists to avoid, and omitting them would fail with an AttributeError
        that says nothing about why.
        """
        if changepoints is not None:
            raise NotImplementedError(
                "changepoints=... (an explicit list) is not supported; "
                "changepoints are placed from n_changepoints and "
                "changepoint_range. Tracked in #15.")
        if mcmc_samples:
            raise NotImplementedError(
                f"mcmc_samples={mcmc_samples} is not supported; this "
                f"implementation is MAP only.")
        if stan_backend is not None:
            raise NotImplementedError(
                f"stan_backend={stan_backend!r} is not supported; there is no "
                f"Stan here -- that is the point of the project.")
        if scaling != 'absmax':
            raise NotImplementedError(
                f"scaling={scaling!r} is not supported; only 'absmax' is "
                f"implemented.")

    def get_parameters(self) -> np.array:
        return self.opt_params
        
    def _normalize_y(self) -> None:
        self.y_scale = np.max(np.abs(self.y))
        self.y_scaled = np.array(self.y / self.y_scale)

    def _future_capacity(self, future_df):
        """(cap_scaled, floor) for a frame passed to predict, or (None, None).

        Logistic growth reads `cap` from whichever frame it is given, so a
        forecast can carry a capacity the history never had.
        """
        if self.growth != 'logistic':
            return None, None
        if 'cap' not in future_df:
            raise ValueError(
                'Capacities must be supplied for logistic growth in column "cap"')
        floor = future_df['floor'].to_numpy(dtype=float) if 'floor' in future_df \
            else np.zeros(len(future_df))
        cap = future_df['cap'].to_numpy(dtype=float)
        if np.any(cap <= floor):
            raise ValueError('cap must be greater than floor (which defaults to 0).')
        return (cap - floor) / self.y_scale, floor

    def _setup_growth(self, df) -> None:
        """Validate the growth mode and read `cap` for logistic growth.

        [fc] setup_dataframe. `floor` defaults to 0 under absmax scaling, which
        is the only scaling here, so `cap_scaled` is `cap / y_scale`.
        """
        if self.growth not in TREND_INDICATORS:
            raise ValueError(
                f"growth={self.growth!r} is not supported; it must be one of "
                f"{', '.join(sorted(TREND_INDICATORS))}")
        if self.growth != 'logistic':
            self.cap_scaled = None
            return

        if 'cap' not in df:
            raise ValueError(
                'Capacities must be supplied for logistic growth in column "cap"')
        floor = df['floor'].to_numpy(dtype=float) if 'floor' in df \
            else np.zeros(len(df))
        cap = df['cap'].to_numpy(dtype=float)
        if np.any(cap <= floor):
            raise ValueError('cap must be greater than floor (which defaults to 0).')

        self.floor = floor
        self.cap_scaled = (cap - floor) / self.y_scale
        self.y_scaled = (self.y - floor) / self.y_scale
    
    def add_holidays(self, holidays, prior_scale=None, mode=None):
        """Register a holidays frame. Returns self, so calls chain.

        [fc] Prophet takes this as a constructor argument; a method keeps the
        validation next to add_seasonality's, which it shares.

        `holidays` is a DataFrame with `holiday` and `ds`, optionally
        `lower_window`/`upper_window` (a window of days around each occurrence,
        each becoming its own column) and `prior_scale` (per holiday, and it
        must be consistent across that holiday's rows).
        """
        if self.opt_params is not None:
            raise RuntimeError(
                "holidays must be added before fitting; this model has already "
                "been fit. Add them to a fresh model.")

        mode = self.seasonality_mode if mode is None else mode
        if mode not in ('additive', 'multiplicative'):
            raise ValueError('holidays_mode must be "additive" or "multiplicative"')
        if prior_scale is not None:
            if not float(prior_scale) > 0:
                raise ValueError("Prior scale must be > 0")
            self.holidays_prior_scale = float(prior_scale)

        self.holidays = validate_holidays_frame(holidays, self.validate_column_name)
        self.holidays_mode = mode
        return self

    def add_country_holidays(self, country_name):
        """Add a country's built-in holidays. Returns self, so calls chain.

        [fc] Prophet.add_country_holidays. These are generated for whichever
        years a frame covers, at fit and at predict alike, so a forecast past
        the end of the history still gets its holidays -- unlike a frame passed
        to add_holidays(), which only contains the dates it lists.

        Only one country at a time, as in Prophet; setting a second replaces
        the first and says so.
        """
        if self.opt_params is not None:
            raise RuntimeError(
                "country holidays must be added before fitting; this model has "
                "already been fit. Add them to a fresh model.")

        # every name the country can produce, checked before it is registered
        # -- a collision only found at fit time would be found after the frame
        # had already been built around it. check_holidays=False so a country
        # may be merged with a hand-written frame naming the same holiday.
        for name in get_holiday_names(country_name):
            self.validate_column_name(name, check_holidays=False)

        if self.country_holidays is not None and self.country_holidays != country_name:
            logger.warning("Changing country holidays from %r to %r.",
                           self.country_holidays, country_name)
        self.country_holidays = country_name
        return self

    def construct_holiday_dataframe(self, dates):
        """The holidays relevant to `dates`, reconciled with what the fit saw.

        [fc] construct_holiday_dataframe. At predict time a holiday the fit
        never saw is dropped -- there is no coefficient for it -- and one the
        fit saw but this frame does not contain is kept with a null `ds`, so
        its column is still present and all zeros.
        """
        if self.holidays is None and self.country_holidays is None:
            return None

        frame = pd.DataFrame(columns=["holiday", "ds"]) if self.holidays is None \
            else self.holidays.copy()
        if self.country_holidays is not None:
            years = sorted({timestamp.year for timestamp in pd.to_datetime(pd.Series(
                np.asarray(dates)))})
            frame = pd.concat([frame, make_holidays_df(years, self.country_holidays)],
                              sort=False, ignore_index=True)

        if self.train_holiday_names is not None:
            frame = frame[frame["holiday"].isin(self.train_holiday_names)]
            missing = [name for name in self.train_holiday_names
                       if name not in set(frame["holiday"])]
            if missing:
                frame = pd.concat([frame, pd.DataFrame({"holiday": missing, "ds": pd.NaT})],
                                  sort=False, ignore_index=True)
        return frame

    def _data_design(self, dates, df, holidays_block=None):
        """(columns, prior_scales, modes) for the holiday and regressor blocks.

        One block because the objective treats them alike: both are columns of
        X with their own entry in `sigmas` and their own mode. They differ only
        in how they are built, which is why the two halves are assembled
        separately and concatenated here.
        """
        if holidays_block is None:
            holidays_block, holiday_scales = self._holiday_design(dates)
        else:
            holiday_scales = self._holiday_prior_scales
        holidays_block = np.empty((len(df), 0)) if holidays_block is None else holidays_block

        holiday_mode = self.holidays_mode or self.seasonality_mode
        modes = [holiday_mode] * holidays_block.shape[1]
        modes += [props["mode"] for props in self.extra_regressors.values()]

        scales = list(holiday_scales) + [props["prior_scale"]
                                         for props in self.extra_regressors.values()]
        columns = np.concatenate([holidays_block, regressor_columns(self.extra_regressors, df)],
                                 axis=1)
        return columns, scales, modes

    def _holiday_design(self, dates):
        """(features, prior_scales) for `dates`, in the fit's column order."""
        frame = self.construct_holiday_dataframe(dates)
        if frame is None:
            return None, []
        return make_holiday_features(dates, frame, self.holidays_prior_scale)[:2]

    def validate_column_name(self, name, check_holidays=True,
                             check_seasonalities=True, check_regressors=True) -> None:
        """Reject a component name that would collide with something else.

        [fc] Prophet.validate_column_name. The reserved list is about the
        *output* frame as much as the model: `predict()` emits a column per
        component, so a seasonality called `trend` or `yhat_lower` would
        overwrite one of its own results.
        """
        if '_delim_' in name:
            raise ValueError('Name cannot contain "_delim_"')
        if name in RESERVED_COLUMN_NAMES:
            raise ValueError(f"Name {name!r} is reserved.")
        if check_holidays and self.holidays is not None and \
                name in set(self.holidays['holiday'].dropna().unique()):
            raise ValueError(f"Name {name!r} already used for a holiday.")
        if check_holidays and self.country_holidays is not None and \
                name in get_holiday_names(self.country_holidays):
            raise ValueError(
                f"Name {name!r} is a holiday name in {self.country_holidays}.")
        if check_seasonalities and name in self.seasonalities:
            raise ValueError(f"Name {name!r} already used for a seasonality.")
        if check_regressors and name in self.extra_regressors:
            raise ValueError(f"Name {name!r} already used for an added regressor.")

    def add_seasonality(self, name, period, fourier_order, prior_scale=None,
                        mode=None, condition_name=None):
        """Register a seasonal component. Returns self, so calls chain.

        [fc] Prophet.add_seasonality. This is the public way into the registry
        that tasks 2-5 built: before it, the only way to fit anything other
        than what the auto rule selects was to assign to `self.seasonalities`.

        A component registered here suppresses the built-in of the same name,
        which is how `add_seasonality('weekly', 7, 10)` asks for a
        higher-resolution weekly term than the default order 3 --
        set_auto_seasonalities leaves a name it finds already registered
        alone.

        `condition_name` names a boolean column that the frames passed to
        fit() and predict() must both carry; rows where it is False have this
        component's features zeroed, so it contributes nothing there.

        `mode` is 'additive' or 'multiplicative'. A multiplicative component
        scales the trend rather than adding to it, so its contribution grows
        with the level of the series.
        """
        if self.opt_params is not None:
            raise RuntimeError(
                "seasonality must be added before fitting; this model has "
                "already been fit. Add it to a fresh model, or register it "
                "before calling fit()/fit_cpp().")

        # Built-in names are exempt: overwriting `weekly` with a different
        # order is a supported thing to want. [fc] passes
        # check_seasonalities=False here, so a custom name may also be
        # re-registered -- the check is for collisions with holidays and
        # regressors, not for re-registration.
        if name not in BUILT_IN_NAMES:
            self.validate_column_name(name, check_seasonalities=False)

        scale = self.seasonality_prior_scale if prior_scale is None else float(prior_scale)
        if scale <= 0:
            raise ValueError("Prior scale must be > 0")
        if int(fourier_order) <= 0:
            raise ValueError("Fourier Order must be > 0")

        mode = self.seasonality_mode if mode is None else mode
        if mode not in ('additive', 'multiplicative'):
            raise ValueError('mode must be "additive" or "multiplicative"')
        if condition_name is not None:
            self.validate_column_name(condition_name)

        entry = seasonality(period, fourier_order, prior_scale=scale, mode=mode,
                            condition_name=condition_name)
        # the same check a fit applies, brought forward to the call site
        check_seasonality_supported({name: entry})

        self.seasonalities[name] = entry
        return self

    def set_auto_seasonalities(self) -> None:
        """Register the built-in seasonalities the history supports.

        [fc] Prophet.set_auto_seasonalities. Runs at the start of a fit, on
        whatever is already registered: a hand-registered component of the same
        name suppresses its built-in, which is how add_seasonality('weekly',
        ...) is meant to win over the automatic one.
        """
        # Prophet rejects a second fit outright, so it never faces this; this
        # model allows one, and a component the previous history supported must
        # not survive into a history that does not. Only what the rule added is
        # cleared -- a hand-registered component is the user's, not ours.
        for name in self._auto_registered:
            self.seasonalities.pop(name, None)
        self._auto_registered = set()

        first, last, min_dt = history_spacing(self.ds)
        span = last - first

        for name, attribute, period, default_order, min_span, max_spacing in AUTO_SEASONALITY_RULES:
            disable = span < min_span or (max_spacing is not None and min_dt >= max_spacing)
            order = parse_seasonality_args(name, getattr(self, attribute), disable,
                                           default_order, self.seasonalities)
            if name == "yearly" and order > 0 and disable:
                logger.warning(UNDER_IDENTIFIED_WARNING)
            if order > 0:
                self.seasonalities[name] = seasonality(
                    period, order, prior_scale=self.seasonality_prior_scale,
                    mode=self.seasonality_mode)
                self._auto_registered.add(name)

    def _build_layout(self) -> None:
        """Fix the parameter-vector layout for this fit, from the changepoint
        count, the registered seasonalities and the holiday columns.

        Replaces the guard that used to reject any non-default changepoint
        count: the layout is no longer a module constant, so both S and K
        follow the model rather than the other way round.
        """
        check_seasonality_supported(self.seasonalities)
        self.layout = ParameterLayout(self.n_changepoints,
                                      seasonality_columns(self.seasonalities),
                                      self._data_column_count)
        # `sigmas` in Stan's data block: one entry per column of the design
        # matrix, so it is fixed by the registry at the same moment the layout
        # is. self.seasonality_prior_scale remains the model-wide default a
        # component inherits when it does not carry its own.
        # [stan] `sigmas` spans every regressor column, seasonal then holiday,
        # in the same order as the design matrix.
        self.sigmas = np.concatenate([seasonality_prior_scales(self.seasonalities),
                                      np.asarray(self._data_prior_scales, dtype=float)])
        # [stan] vector[K] s_m / s_a: which columns multiply the trend and
        # which add to it. Every holiday column shares holidays_mode.
        self.s_m = np.concatenate([
            seasonality_modes(self.seasonalities),
            np.array([1.0 if mode == "multiplicative" else 0.0
                      for mode in self._data_modes])])
        self.s_a = 1.0 - self.s_m
        self._multiplicative = bool(np.any(self.s_m))

    def _generate_change_points(self) -> None:
        """Changepoints spaced uniformly in *scaled time* over the first
        changepoint_range of history.

        KNOWN DIVERGENCE from [fc] set_changepoints, which spaces them over
        uniformly-spaced row *indices* of the first 80% of history:

            np.linspace(0, hist_size - 1, n_changepoints + 1).round().astype(int)

        and then takes the ds values at those rows. For regularly-spaced daily
        data the two agree; for irregular or gappy series they diverge, since
        index-spacing follows observation density and time-spacing does not.
        Left as-is deliberately -- see the follow-up issue.
        """
        max_t_scaled = np.max(self.t_scaled)
        self.t_change = np.linspace(0, self.changepoint_range * max_t_scaled, self.n_changepoints + 1)[1:]

        
    def _design_matrices(self):
        """The changepoint indicator A (T x S) and the Fourier design matrix
        (T x K).

        Both depend only on t_scaled and t_change, so they are constant
        for a whole fit -- yet rebuilding them was 57% of every objective
        evaluation (issue #28). fit() computes them once and threads them
        through; callers that pass nothing still get correct results, which is
        what keeps the objective usable on its own.
        """
        A = (self.t_scaled[:, None] >= self.t_change) * 1
        # [fc] make_all_seasonality_features appends holiday columns after the
        # seasonal ones; `beta` and `sigmas` follow that order.
        x = seasonality_design_matrix(self.t_seasonality, self.seasonalities,
                                      self.condition_masks)
        if self._data_column_count:
            x = np.concatenate([x, self._data_columns], axis=1)
        return A, x

    def _minus_log_posterior(self, params: np.array, include_l1_prior: bool=True, design=None) -> float:
        k, m, delta, sigma_obs, beta = extract_params(params, self.layout)

        A, x = design if design is not None else self._design_matrices()

        # trend component. Linear keeps its own expression rather than going
        # through the Jacobian form: it is the common case, and its fits are
        # sensitive to the last bits (see the note on the additive branch).
        if self.growth == 'logistic':
            g, trend_jacobian = logistic_trend_and_jacobian(
                k, m, delta, self.t_scaled, self.cap_scaled, A, self.t_change)
        elif self.growth == 'flat':
            # [stan] flat_trend: rep_vector(m, T). k and delta remain
            # parameters and keep their priors, but the likelihood does not see
            # them, so both are driven to zero.
            trend_jacobian = None
            g = np.full(self.T, m)
        else:
            trend_jacobian = None
            gamma = -self.t_change * delta
            g = (k + np.dot(A, delta)) * self.t_scaled + (m + np.dot(A, gamma))

        # [stan] y ~ normal_id_glm(X_sa, trend .* (1 + X_sm * beta), beta, ...)
        # -- additive columns add to the trend, multiplicative ones scale it.
        y_pred = g + np.dot(x, beta) if not self._multiplicative else \
            g * (1.0 + np.dot(x * self.s_m, beta)) + np.dot(x * self.s_a, beta)
        y_true = self.y_scaled

        # T*log(sigma_obs) is the Gaussian normalizing constant -- it can't be
        # dropped now that sigma_obs is itself a parameter being optimized.
        minus_log_posterior = self.T * np.log(sigma_obs) + \
                      np.sum((y_true - y_pred)**2) / (2*sigma_obs**2) + \
                      sigma_obs**2 / (2*SIGMA_OBS_PRIOR_SCALE**2) + \
                      k**2 / (2*self.sigma_k**2) + \
                      m**2 / (2*self.sigma_m**2) + \
                      np.sum(beta**2 / (2 * self.sigmas**2))

        if include_l1_prior:
            minus_log_posterior += np.sum(np.abs(delta)) / self.tau

        return minus_log_posterior

    def _gradient(self, params: np.array, include_l1_prior: bool=True, design=None) -> np.array:
        k, m, delta, sigma_obs, beta = extract_params(params, self.layout)

        A, x = design if design is not None else self._design_matrices()

        # trend component. Linear keeps its own expression rather than going
        # through the Jacobian form: it is the common case, and its fits are
        # sensitive to the last bits (see the note on the additive branch).
        if self.growth == 'logistic':
            g, trend_jacobian = logistic_trend_and_jacobian(
                k, m, delta, self.t_scaled, self.cap_scaled, A, self.t_change)
        elif self.growth == 'flat':
            # [stan] flat_trend: rep_vector(m, T). k and delta remain
            # parameters and keep their priors, but the likelihood does not see
            # them, so both are driven to zero.
            trend_jacobian = None
            g = np.full(self.T, m)
        else:
            trend_jacobian = None
            gamma = -self.t_change * delta
            g = (k + np.dot(A, delta)) * self.t_scaled + (m + np.dot(A, gamma))

        # [stan] y ~ normal_id_glm(X_sa, trend .* (1 + X_sm * beta), beta, ...).
        # The all-additive case takes its own branch rather than multiplying by
        # a vector of ones. Not only for speed: `y - g - s` and `y - (g*1 + s)`
        # differ in the last bits, and on this objective's flat directions that
        # was enough to move the scipy path to a point 2.96 nats worse.
        if self._multiplicative:
            multiplier = 1.0 + np.dot(x * self.s_m, beta)
            r = self.y_scaled - (g * multiplier + np.dot(x * self.s_a, beta))
        else:
            multiplier = None
            r = self.y_scaled - g - np.dot(x, beta)

        # Every trend block picks up the multiplier, since d(yhat)/d(a trend
        # parameter) = d(g)/d(that parameter) * multiplier. beta's picks up the
        # trend on its multiplicative columns. All of them reduce to the
        # additive forms when s_m is zero, because the multiplier is then 1.
        r_scaled = r if multiplier is None else r * multiplier
        if self.growth == 'flat':
            # d(trend)/dk and d(trend)/d(delta) are zero, so those blocks carry
            # their priors alone
            dk = np.array([k / self.sigma_k**2])
            dm = np.array([-np.sum(r_scaled) / sigma_obs**2 + m / self.sigma_m**2])
            ddelta = np.zeros(len(delta))
        elif trend_jacobian is None:
            dk = np.array([-np.sum(r_scaled * self.t_scaled) / sigma_obs**2 + k / self.sigma_k**2])
            dm = np.array([-np.sum(r_scaled) / sigma_obs**2 + m / self.sigma_m**2])
            ddelta = -np.sum(r_scaled[:, None] * (self.t_scaled[:, None] - self.t_change) * A, axis=0) / sigma_obs**2
        else:
            # one contraction for all three blocks: the Jacobian already holds
            # d(trend)/d(k, m, delta), including gamma's recursion
            d_trend = -np.dot(r_scaled, trend_jacobian) / sigma_obs**2
            dk = np.array([d_trend[0] + k / self.sigma_k**2])
            dm = np.array([d_trend[1] + m / self.sigma_m**2])
            ddelta = d_trend[2:]
        dsigma_obs = np.array([self.T / sigma_obs - np.sum(r**2) / sigma_obs**3 + sigma_obs / SIGMA_OBS_PRIOR_SCALE**2])
        dbeta = (-np.dot(r, x) / sigma_obs**2 if multiplier is None else
                 -(np.dot(r * g, x * self.s_m) + np.dot(r, x * self.s_a)) / sigma_obs**2) \
                + beta / self.sigmas**2

        if include_l1_prior:
            ddelta = ddelta + np.sign(delta) / self.tau

        gradient = np.concatenate([dk, dm, ddelta, dsigma_obs, dbeta])

        return gradient

    def _minus_log_posteriorAndGradient(self, params: np.array, include_l1_prior: bool=True, design=None) -> Tuple[float, np.array]:
        k, m, delta, sigma_obs, beta = extract_params(params, self.layout)

        A, x = design if design is not None else self._design_matrices()

        # trend component. Linear keeps its own expression rather than going
        # through the Jacobian form: it is the common case, and its fits are
        # sensitive to the last bits (see the note on the additive branch).
        if self.growth == 'logistic':
            g, trend_jacobian = logistic_trend_and_jacobian(
                k, m, delta, self.t_scaled, self.cap_scaled, A, self.t_change)
        elif self.growth == 'flat':
            # [stan] flat_trend: rep_vector(m, T). k and delta remain
            # parameters and keep their priors, but the likelihood does not see
            # them, so both are driven to zero.
            trend_jacobian = None
            g = np.full(self.T, m)
        else:
            trend_jacobian = None
            gamma = -self.t_change * delta
            g = (k + np.dot(A, delta)) * self.t_scaled + (m + np.dot(A, gamma))

        # [stan] y ~ normal_id_glm(X_sa, trend .* (1 + X_sm * beta), beta, ...).
        # The all-additive case takes its own branch rather than multiplying by
        # a vector of ones. Not only for speed: `y - g - s` and `y - (g*1 + s)`
        # differ in the last bits, and on this objective's flat directions that
        # was enough to move the scipy path to a point 2.96 nats worse.
        if self._multiplicative:
            multiplier = 1.0 + np.dot(x * self.s_m, beta)
            r = self.y_scaled - (g * multiplier + np.dot(x * self.s_a, beta))
        else:
            multiplier = None
            r = self.y_scaled - g - np.dot(x, beta)

        minus_log_posterior = self.T * np.log(sigma_obs) + \
                      np.sum(r**2) / (2*sigma_obs**2) + \
                      sigma_obs**2 / (2*SIGMA_OBS_PRIOR_SCALE**2) + \
                      k**2 / (2*self.sigma_k**2) + \
                      m**2 / (2*self.sigma_m**2) + \
                      np.sum(beta**2 / (2 * self.sigmas**2))

        # Every trend block picks up the multiplier, since d(yhat)/d(a trend
        # parameter) = d(g)/d(that parameter) * multiplier. beta's picks up the
        # trend on its multiplicative columns. All of them reduce to the
        # additive forms when s_m is zero, because the multiplier is then 1.
        r_scaled = r if multiplier is None else r * multiplier
        if self.growth == 'flat':
            # d(trend)/dk and d(trend)/d(delta) are zero, so those blocks carry
            # their priors alone
            dk = np.array([k / self.sigma_k**2])
            dm = np.array([-np.sum(r_scaled) / sigma_obs**2 + m / self.sigma_m**2])
            ddelta = np.zeros(len(delta))
        elif trend_jacobian is None:
            dk = np.array([-np.sum(r_scaled * self.t_scaled) / sigma_obs**2 + k / self.sigma_k**2])
            dm = np.array([-np.sum(r_scaled) / sigma_obs**2 + m / self.sigma_m**2])
            ddelta = -np.sum(r_scaled[:, None] * (self.t_scaled[:, None] - self.t_change) * A, axis=0) / sigma_obs**2
        else:
            # one contraction for all three blocks: the Jacobian already holds
            # d(trend)/d(k, m, delta), including gamma's recursion
            d_trend = -np.dot(r_scaled, trend_jacobian) / sigma_obs**2
            dk = np.array([d_trend[0] + k / self.sigma_k**2])
            dm = np.array([d_trend[1] + m / self.sigma_m**2])
            ddelta = d_trend[2:]
        dsigma_obs = np.array([self.T / sigma_obs - np.sum(r**2) / sigma_obs**3 + sigma_obs / SIGMA_OBS_PRIOR_SCALE**2])
        dbeta = (-np.dot(r, x) / sigma_obs**2 if multiplier is None else
                 -(np.dot(r * g, x * self.s_m) + np.dot(r, x * self.s_a)) / sigma_obs**2) \
                + beta / self.sigmas**2

        if include_l1_prior:
            minus_log_posterior += np.sum(np.abs(delta)) / self.tau
            ddelta = ddelta + np.sign(delta) / self.tau

        gradient = np.concatenate([dk, dm, ddelta, dsigma_obs, dbeta])

        return minus_log_posterior, gradient

    # --- split-space (delta = delta_pos - delta_neg) wrappers --------------
    # These are what fit() optimizes over. They evaluate the posterior without
    # its L1 term and add (delta_pos + delta_neg)/tau instead, which is the
    # same value but differentiable -- see canonical_to_split for why.

    def _split_l1_penalty(self, z):
        n_delta = self.layout.n_changepoints
        return np.sum(z[2:2 + 2 * n_delta]) / self.tau

    def _split_minus_log_posterior(self, z: np.array, design=None) -> float:
        smooth = self._minus_log_posterior(split_to_canonical(z, self.layout.n_changepoints),
                                           include_l1_prior=False, design=design)
        return smooth + self._split_l1_penalty(z)

    def _split_gradient(self, z: np.array, design=None) -> np.array:
        return self._canonical_gradient_to_split(
            self._gradient(split_to_canonical(z, self.layout.n_changepoints),
                           include_l1_prior=False, design=design))

    def _split_minus_log_posteriorAndGradient(self, z: np.array, design=None) -> Tuple[float, np.array]:
        smooth, gradient = self._minus_log_posteriorAndGradient(
            split_to_canonical(z, self.layout.n_changepoints), include_l1_prior=False, design=design)
        return smooth + self._split_l1_penalty(z), self._canonical_gradient_to_split(gradient)

    def _canonical_gradient_to_split(self, gradient):
        """d/d(delta_pos) = d/d(delta) + 1/tau, d/d(delta_neg) = -d/d(delta) + 1/tau."""
        ddelta = gradient[self.layout.delta]
        return np.concatenate((
            gradient[:2],
            ddelta + 1 / self.tau,
            -ddelta + 1 / self.tau,
            [gradient[self.layout.sigma_obs_idx]],
            gradient[self.layout.beta],
        ))

    def fit(self, df: pd.DataFrame, analytic: bool=False, use_combined: bool=False, optimizer: str='L-BFGS-B',
            initial_params: dict=None, fixed_sigma_obs: float=None) -> Tuple[float, float, np.array, np.array]:
        if analytic and use_combined:
            raise ValueError("Both 'analytic' and 'use_combined' cannot be True at the same time.")

        self.y = df['y'].values

        if df['ds'].dtype != 'datetime64[ns]':
            self.ds = pd.to_datetime(df['ds'])
        else:
            self.ds = df['ds']

        self.t_scaled = np.array((self.ds - self.ds.min()) / (self.ds.max() - self.ds.min()))
        self.T = df.shape[0]

        # Calculate the scale period coefficient
        self.t_seasonality = seasonal_time(self.ds)

        self._normalize_y()
        self._setup_growth(df)
        self.set_auto_seasonalities()
        # after selection, since a conditioned component may have been added by
        # hand while an auto-selected one never carries a condition
        self.condition_masks = condition_masks(self.seasonalities, df)
        holidays_block, self._holiday_prior_scales = self._holiday_design(self.ds)
        self._holiday_columns = 0 if holidays_block is None else holidays_block.shape[1]
        # recorded after the features are built, so predict presents the same
        # columns even for a holiday that never lands in the future frame
        if self.train_holiday_names is None and self._holiday_columns:
            self.train_holiday_names = list(make_holiday_features(
                self.ds, self.construct_holiday_dataframe(self.ds), self.holidays_prior_scale)[2])

        # standardization is fitted here, on the history, and reapplied
        # unchanged at predict time. [fc] initialize_scales.
        for name, props in self.extra_regressors.items():
            if name not in df:
                raise ValueError(f"Regressor {name!r} missing from dataframe")
            props["mu"], props["std"] = regressor_standardization(
                df[name], props["standardize"])
        self._data_columns, self._data_prior_scales, self._data_modes = \
            self._data_design(self.ds, df, holidays_block)
        self._data_column_count = self._data_columns.shape[1]
        self._build_layout()
        self._generate_change_points()

        # [fc] calculate_initial_params: k/m from linear_growth_init, delta and
        # beta at zero, sigma_obs at 1.0. Prophet passes these to Stan
        # explicitly, so Stan's random init is never used.
        if self.growth == 'logistic':
            k_init, m_init = logistic_growth_init(self.t_scaled, self.y_scaled,
                                                  self.cap_scaled)
        elif self.growth == 'flat':
            k_init, m_init = flat_growth_init(self.y_scaled)
        else:
            k_init, m_init = linear_growth_init(self.t_scaled, self.y_scaled)
        initial_params_dict = {
            'k': k_init,
            'm': m_init,
            'delta': np.zeros(self.layout.n_changepoints),
            'sigma_obs': SIGMA_OBS_INIT,
            'beta': np.zeros(self.layout.n_regressor_columns),
        }
        if initial_params is not None:
            initial_params_dict.update(initial_params)
        if fixed_sigma_obs is not None:
            initial_params_dict['sigma_obs'] = fixed_sigma_obs

        loss_over_iterations = []

        # Built once here, not rebuilt on every evaluation (#28).
        design = self._design_matrices()

        def callback(z):
            fobj = self._minus_log_posterior(split_to_canonical(z, self.layout.n_changepoints), design=design)
            loss_over_iterations.append(fobj)

        initial_params_array = from_dict_to_array(initial_params_dict, self.layout)

        # sigma_obs must stay positive, mirroring Stan's `real<lower=0> sigma_obs`.
        # fixed_sigma_obs collapses that bound to a single point, pinning sigma_obs
        # for parity with fit_cpp()'s compiled optimizer, which never estimates it.
        sigma_obs_bounds = (fixed_sigma_obs, fixed_sigma_obs) if fixed_sigma_obs is not None else (1e-6, None)
        n_delta = self.layout.n_changepoints
        # Split-space bounds: k, m free; delta_pos/delta_neg >= 0; then sigma_obs, beta
        bounds = [(None, None)] * 2 + [(0, None)] * (2 * n_delta) + [sigma_obs_bounds] + \
                 [(None, None)] * self.layout.n_regressor_columns

        z0 = canonical_to_split(initial_params_array, self.layout)

        if use_combined:
            objective = lambda z: self._split_minus_log_posteriorAndGradient(z, design=design)
            jac = True
        elif analytic:
            objective = lambda z: self._split_minus_log_posterior(z, design=design)
            jac = lambda z: self._split_gradient(z, design=design)
        else:
            objective = lambda z: self._split_minus_log_posterior(z, design=design)
            jac = None

        options = {'maxiter': STAN_MAX_ITERATIONS}
        if optimizer == 'L-BFGS-B':
            # Stan's iteration cap applies directly. Its tolerances do not
            # transfer as cleanly here as they do in the C++ core, for two
            # measured reasons -- both consequences of this path optimizing the
            # split reformulation rather than Stan's parameterization:
            #
            # gtol is disabled rather than set to Stan's tol_grad. scipy tests
            # the inf-norm of the *projected* gradient, Stan the 2-norm of the
            # full gradient. On the split problem most delta_pos/delta_neg sit
            # at their zero bound, so the projected norm is far smaller than
            # the real one and tol_grad=1e-8 fires at iteration 147 on the
            # 2905-point series, 1.1e-3 short in loss.
            #
            # ftol stays tighter than Stan's tol_rel_obj * eps = 2.22e-12. The
            # split space has long shallow ridges where per-iteration progress
            # falls below that while the fit is still 1.1e-3 from the optimum
            # (measured at T=1000 and T=2905); Stan's own parameterization does
            # not stall there, and the C++ core, which uses it, converges
            # normally under the real tolerance. Loosening this to match Stan
            # numerically would mean a worse fit than Prophet produces, not a
            # closer one.
            options.update({
                'ftol': 1e-16,
                'gtol': 0.0,
                'maxfun': STAN_MAX_ITERATIONS * 10,
            })

        opt_params = minimize(objective,
                        z0,
                        method=optimizer,
                        bounds=bounds,
                        options=options,
                        callback=callback,
                        jac=jac)

        self.opt = opt_params
        self.opt_params = split_to_canonical(opt_params.x, self.layout.n_changepoints)
        self.sigma_obs = self.opt_params[self.layout.sigma_obs_idx]
        self._fitted_with_cpp = False
        self._fit_regressor_models(df)
        self.loss_over_iterations = loss_over_iterations
    
    def fit_cpp(self, df: pd.DataFrame, initial_params: dict=None, lib_path: str=None, verbose: bool=False) -> Tuple[float, float, np.array, np.array]:
        """Fit via the compiled C++ core (see load_cpp_module for how it's found)."""
        self.y = df['y'].values

        if df['ds'].dtype != 'datetime64[ns]':
            self.ds = pd.to_datetime(df['ds'])
        else:
            self.ds = df['ds']

        self.t_scaled = np.array((self.ds - self.ds.min()) / (self.ds.max() - self.ds.min()))
        self.T = df.shape[0]

        self.t_seasonality = seasonal_time(self.ds)
        self._normalize_y()
        self._setup_growth(df)
        self.set_auto_seasonalities()
        # after selection, since a conditioned component may have been added by
        # hand while an auto-selected one never carries a condition
        self.condition_masks = condition_masks(self.seasonalities, df)
        holidays_block, self._holiday_prior_scales = self._holiday_design(self.ds)
        self._holiday_columns = 0 if holidays_block is None else holidays_block.shape[1]
        # recorded after the features are built, so predict presents the same
        # columns even for a holiday that never lands in the future frame
        if self.train_holiday_names is None and self._holiday_columns:
            self.train_holiday_names = list(make_holiday_features(
                self.ds, self.construct_holiday_dataframe(self.ds), self.holidays_prior_scale)[2])

        # standardization is fitted here, on the history, and reapplied
        # unchanged at predict time. [fc] initialize_scales.
        for name, props in self.extra_regressors.items():
            if name not in df:
                raise ValueError(f"Regressor {name!r} missing from dataframe")
            props["mu"], props["std"] = regressor_standardization(
                df[name], props["standardize"])
        self._data_columns, self._data_prior_scales, self._data_modes = \
            self._data_design(self.ds, df, holidays_block)
        self._data_column_count = self._data_columns.shape[1]
        self._build_layout()
        self._generate_change_points()

        # Same deterministic initialization as fit(), so both fit paths start
        # from the same point. [fc] calculate_initial_params.
        #
        # This used to draw init_r * N(0, 1) with init_r=2.0, described as
        # "STAN initialization" -- but Prophet always passes explicit inits, so
        # Stan's random default is never reached. The draw also came from an
        # unseeded RNG, making fit_cpp() non-reproducible run to run.
        if self.growth == 'logistic':
            k_init, m_init = logistic_growth_init(self.t_scaled, self.y_scaled,
                                                  self.cap_scaled)
        elif self.growth == 'flat':
            k_init, m_init = flat_growth_init(self.y_scaled)
        else:
            k_init, m_init = linear_growth_init(self.t_scaled, self.y_scaled)
        defaults = {
            'k': k_init,
            'm': m_init,
            'delta': np.zeros(self.layout.n_changepoints),
            'beta': np.zeros(self.layout.n_regressor_columns),
        }
        if initial_params is not None:
            defaults.update(initial_params)

        # The compiled optimizer's layout is [k, m, delta(S), beta(K), zeta],
        # with zeta = log(sigma_obs) last -- see cpp_to_canonical. Estimating
        # the log keeps sigma_obs positive without box constraints, which
        # liblbfgs does not have.
        zeta_init = np.log(defaults.get('sigma_obs', SIGMA_OBS_INIT))
        params = np.concatenate(([defaults['k']], [defaults['m']],
                                 defaults['delta'], defaults['beta'], [zeta_init]))

        cpp = load_cpp_module(lib_path)
        result = cpp.optimize(
            params=params,
            t_scaled=self.t_scaled,
            t_change=self.t_change,
            t_seasonality=self.t_seasonality,
            y_scaled=self.y_scaled,
            sigma_obs_prior_scale=SIGMA_OBS_PRIOR_SCALE,
            sigma_k=self.sigma_k,
            sigma_m=self.sigma_m,
            sigmas=self.sigmas,
            s_m=self.s_m,
            cap_scaled=(self.cap_scaled if self.growth == 'logistic'
                        else np.empty(0)),
            trend_indicator=TREND_INDICATORS[self.growth],
            tau=self.tau,
            fourier_orders=[props["fourier_order"] for props in self.seasonalities.values()],
            seasonality_periods=[props["period"] for props in self.seasonalities.values()],
            seasonality_conditions=condition_matrix(self.seasonalities,
                                                    self.condition_masks, self.T),
            data_columns=(self._data_columns if self._data_column_count
                              else np.empty((self.T, 0))),
            verbose=verbose,
        )

        # Mirrors fit(), so both paths expose a comparable loss trajectory --
        # with one caveat: fit()'s is per iteration, while the C++ core's is a
        # monotone envelope over objective evaluations, since LBFGSpp offers no
        # per-iteration hook. Both decrease monotonically to the same value.
        self.loss_over_iterations = list(result.loss_trace)
        self.opt = result
        self.opt_status = result.status
        self.opt_status_message = result.status_message

        if not np.all(np.isfinite(result.params)):
            raise RuntimeError(
                f"the C++ optimizer returned non-finite parameters "
                f"(status {result.status}: {result.status_message})")

        # Back to the canonical (k, m, delta, sigma_obs, beta) layout, so
        # predict()/trend_forecast_uncertainty() work the same regardless of
        # which fit method produced opt_params.
        self.opt_params = cpp_to_canonical(result.params, self.layout)
        self.sigma_obs = self.opt_params[self.layout.sigma_obs_idx]
        self._fitted_with_cpp = True
        self._fit_lib_path = lib_path
        self._fit_regressor_models(df)

        # Return whatever values are necessary
        return -1
        
        
    def add_regressor(self, name, prior_scale=None, standardize='auto', mode=None,
                      regressor_predictor=None):
        """Register an extra regressor. Returns self, so calls chain.

        [fc] Prophet.add_regressor. The column is read from the dataframe
        passed to fit() and predict() by `name`, rather than being handed over
        as a series -- which is what the stub this replaces took (#33), and why
        it could not have worked: a series carries no way to produce the future
        values predict() needs.

        `prior_scale` defaults to `holidays_prior_scale`, not to
        `seasonality_prior_scale`. That looks like a mistake in the original
        and is not: [fc] `prior_scale = float(self.holidays_prior_scale)`.

        `standardize='auto'` standardizes unless the column is binary. The mean
        and standard deviation are fitted on the history and reapplied
        unchanged at predict time, so a future frame with a different spread
        does not rescale the coefficient out from under itself.
        """
        if self.opt_params is not None:
            raise RuntimeError(
                "regressors must be added before fitting; this model has "
                "already been fit. Add them to a fresh model.")

        self.validate_column_name(name, check_regressors=False)

        prior_scale = self.holidays_prior_scale if prior_scale is None else float(prior_scale)
        mode = self.seasonality_mode if mode is None else mode
        if prior_scale <= 0:
            raise ValueError("Prior scale must be > 0")
        if mode not in ('additive', 'multiplicative'):
            raise ValueError("mode must be 'additive' or 'multiplicative'")

        # [fc] a truthy non-dict means "default settings"; a dict is the
        # nested model's configuration.
        predictor_spec = None
        if regressor_predictor:
            predictor_spec = dict(regressor_predictor) if isinstance(regressor_predictor, dict) else {}
            # [fc] the spec goes to the constructor, which rejects what it does
            # not know. This used to run against a hand-kept whitelist, because
            # the constructor took no arguments (#16 task 14c, #52).
            CustomProphet(**predictor_spec)

        # mu and std are placeholders until a fit measures them on the history
        self.extra_regressors[name] = {"prior_scale": prior_scale,
                                       "standardize": standardize,
                                       "mu": 0.0, "std": 1.0, "mode": mode,
                                       "predictor_spec": predictor_spec,
                                       "predictor": None}
        return self

    def _fit_regressor_models(self, df) -> None:
        """Fit a nested model per regressor that asked for one.

        [fc] _fit_regressor_models. The nested model is fitted on the raw
        regressor values, not the standardized ones -- it forecasts the column
        the user supplies, and this model standardizes whatever comes back.

        It is fitted the same way this model was, compiled path included, so a
        regressor predictor does not quietly fall back to the slow path.
        """
        if self.extra_regressors:
            self._regressor_history = df[["ds", *self.extra_regressors]].copy()
            self._regressor_history["ds"] = pd.to_datetime(self._regressor_history["ds"])

        for name, props in self.extra_regressors.items():
            if props.get("predictor_spec") is None:
                continue

            regressor_df = df[["ds", name]].copy()
            regressor_df = regressor_df[regressor_df[name].notnull()]
            if regressor_df.shape[0] < 2:
                raise ValueError(
                    f"Not enough data to fit regressor model for {name!r}.")
            regressor_df = regressor_df.rename(columns={name: "y"})

            predictor = CustomProphet(**props["predictor_spec"])
            predictor._regressor_name = name          # marker, [fc]

            logger.info("Fitting regressor model %r with %d observations",
                        name, regressor_df.shape[0])
            if self._fitted_with_cpp:
                predictor.fit_cpp(regressor_df, lib_path=self._fit_lib_path)
            else:
                predictor.fit(regressor_df)
            props["predictor"] = predictor

    def _ensure_regressor_values(self, future_df):
        """Fill future regressor values from their nested models.

        [fc] _ensure_regressor_values. "Future" is any row past the end of the
        history, which is where the caller has nothing to supply. Rows inside
        the history keep whatever they were given.

        Returns a copy: predict() must not write into the frame it is handed.
        """
        if not any(props.get("predictor") is not None
                   for props in self.extra_regressors.values()):
            return future_df

        filled = future_df.copy()
        last_history_date = pd.to_datetime(self.ds).max()
        is_future = pd.to_datetime(filled["ds"]) > last_history_date

        for name, props in self.extra_regressors.items():
            predictor = props.get("predictor")
            if predictor is None:
                continue
            if name not in filled:
                filled[name] = np.nan

            # [fc] history rows the caller left out come back from the fit's
            # own values, not from the nested model -- the model is there to
            # forecast, not to re-explain what was already observed.
            missing_history = (~is_future) & filled[name].isna()
            if missing_history.any() and self._regressor_history is not None:
                lookup = self._regressor_history.set_index("ds")[name]
                filled.loc[missing_history, name] = lookup.reindex(
                    pd.to_datetime(filled.loc[missing_history, "ds"])).to_numpy()

            if is_future.any():
                forecast = predictor.predict(
                    pd.DataFrame({"ds": filled.loc[is_future, "ds"].to_numpy()}))
                filled.loc[is_future, name] = forecast["yhat"].to_numpy()
        return filled

    

    def make_future_dataframe(self, periods, include_history=True):
        last_date = pd.to_datetime(self.ds.max())  # Ensure last_date is a datetime object
        future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, periods + 1)]
        future_dates_df = pd.DataFrame(future_dates, columns=['ds'])

        if include_history:
            history_dates_df = pd.DataFrame(self.ds, columns=['ds'])
            future_df = pd.concat([history_dates_df, future_dates_df], ignore_index=True)
        else:
            future_df = future_dates_df

        return future_df
    
    def _regressor_draws(self, future_df, n_samples):
        """{name: (n_samples, T)} of raw regressor values, for the regressors
        that have a nested model.

        [fc] _prepare_regressors_for_predict collects `predictive_samples` from
        each nested model and the sampler swaps that column in per draw. The
        draws are the nested model's own forecast distribution, so a regressor
        this model is unsure about widens the interval rather than entering as
        a point estimate (#16 task 14a).
        """
        drawn = {}
        last_history_date = pd.to_datetime(self.ds).max()
        is_future = (pd.to_datetime(future_df["ds"]) > last_history_date).to_numpy()
        for name, props in self.extra_regressors.items():
            predictor = props.get("predictor")
            if predictor is None:
                continue
            # rows inside the history are observed, so only the future varies
            columns = np.tile(future_df[name].to_numpy(dtype=float), (n_samples, 1))
            if is_future.any():
                future_frame = pd.DataFrame({"ds": future_df.loc[is_future, "ds"].to_numpy()})
                columns[:, is_future] = predictor._sample_yhat(future_frame, n_samples)
            drawn[name] = columns
        return drawn

    def _sample_yhat(self, future_df, n_samples):
        """`n_samples` x T draws of this model's own yhat, for a nested model
        to hand its parent."""
        prepared = self._ensure_regressor_values(future_df).copy()
        prepared['t_scaled'] = ((pd.to_datetime(prepared['ds']) - self.ds.min())
                                / (self.ds.max() - self.ds.min()))
        cap_scaled, floor = self._future_capacity(prepared)
        _, _, _, yhat_draws = self._forecast_draws(prepared, cap_scaled, floor, n_samples)
        return yhat_draws

    def _forecast_draws(self, future_df, cap_scaled, floor, n_samples):
        """(x, trend_draws, seasonality, yhat_draws) for a prepared frame.

        `yhat` is drawn rather than derived from the trend band, which is what
        lets two things enter that could not before: a regressor's own forecast
        uncertainty (#16 task 14a) and the observation noise (#63), the latter
        being the term that dominates the interval.
        """
        _k, _m, _delta, sigma_obs, beta = extract_params(self.opt_params, self.layout)
        t_scaled = future_df['t_scaled'].values

        x = seasonality_design_matrix(seasonal_time(future_df['ds']), self.seasonalities,
                                      condition_masks(self.seasonalities, future_df))
        if self._data_column_count:
            x = np.concatenate([x, self._data_design(future_df['ds'], future_df)[0]], axis=1)

        trend_draws = self._sample_trends(t_scaled, cap_scaled, floor, n_samples)
        regressor_draws = self._regressor_draws(future_df, n_samples)

        # the regressor columns are the last of the design matrix, in the order
        # they were registered
        positions = {name: x.shape[1] - len(self.extra_regressors) + i
                     for i, name in enumerate(self.extra_regressors)}

        yhat_draws = np.empty_like(trend_draws)
        x_draw = x
        for i in range(n_samples):
            if regressor_draws:
                x_draw = x.copy()
                for name, values in regressor_draws.items():
                    props = self.extra_regressors[name]
                    x_draw[:, positions[name]] = (values[i] - props["mu"]) / props["std"]
            multiplier = 1.0 + x_draw.dot(self.s_m * beta) if self._multiplicative else 1.0
            seasonal = (x_draw.dot(self.s_a * beta) if self._multiplicative
                        else x_draw.dot(beta))
            # [fc] sample_model adds normal(0, sigma_obs) per draw; without it
            # the interval is the trend's alone and ~18x too narrow (#63)
            noise = self.rng.normal(0.0, sigma_obs * self.y_scale, len(t_scaled))
            yhat_draws[i] = trend_draws[i] * multiplier + seasonal * self.y_scale + noise

        multiplier = 1.0 + x.dot(self.s_m * beta) if self._multiplicative else 1.0
        seasonality = x.dot(self.s_a * beta) if self._multiplicative else x.dot(beta)
        return x, trend_draws, (multiplier, seasonality), yhat_draws

    def _quantiles(self, draws):
        """[fc] predict_uncertainty: a centred interval, so the edges are
        (1 -+ interval_width) / 2."""
        lower = 100 * (1.0 - self.interval_width) / 2
        upper = 100 * (1.0 + self.interval_width) / 2
        return np.percentile(draws, [lower, upper], axis=0)

    def _sample_trends(self, t_scaled, cap_scaled, floor, n_samples):
        """`n_samples` x T trend draws, in the series' own units.

        [fc] sample_predictive_trend. New changepoints come from a Poisson
        process on `(1, T]`, so they land strictly past the end of the history
        and there are none when the frame does not reach past it (#58).
        """
        k, m, delta, _sigma_obs, _beta = extract_params(self.opt_params, self.layout)
        horizon_scaled = float(np.max(t_scaled))
        n_changepoints = len(self.t_change)
        # [fc] `+ 1e-8`: a fit with no active changepoints gives mean|delta| of
        # exactly zero, and Laplace(0, 0) is degenerate -- every draw would
        # return the same trend and the band would be identically zero rather
        # than narrow.
        lambda_mle = float(np.abs(delta).mean()) + 1e-8

        draws = []
        for _ in range(n_samples):
            if horizon_scaled > 1.0:
                n_new = self.rng.poisson(n_changepoints * (horizon_scaled - 1.0))
            else:
                n_new = 0
            new_change_points = np.sort(
                1.0 + self.rng.random(n_new) * (horizon_scaled - 1.0))
            new_delta = self.rng.laplace(0, lambda_mle, n_new)

            draws.append(predict_trend(
                k, m, np.concatenate((delta, new_delta)),
                np.concatenate((self.t_change, new_change_points)),
                t_scaled, self.y_scale, cap_scaled, floor, self.growth))
        return np.array(draws)

    def trend_forecast_uncertainty(self, horizon=30, n_samples=None,
                                   t_scaled=None, cap_scaled=None, floor=None):
        """Quantiles of the trend under future changepoints drawn from the
        fitted rate distribution.

        [fc] sample_predictive_trend. New changepoints come from a Poisson
        process on `(1, T]`, where `T` is the largest scaled time in the frame
        being forecast -- so they land strictly past the end of the history,
        and there are none at all when the frame does not extend past it.

        This previously read a per-point probability of `n_changepoints / T`
        and applied it across the whole grid, history included, so most of the
        sampled changepoints rewrote the fitted history before anything was
        extrapolated from it. The band came out ~170x Prophet's (#58).

        `t_scaled`, `cap_scaled` and `floor` come from the caller: predict()
        passes the grid and capacities it already built, since the capacity is
        per row and the frame this method builds for itself carries only `ds`.
        """
        k, m, delta, _sigma_obs, beta = extract_params(self.opt_params, self.layout)
        n_samples = self.uncertainty_samples if n_samples is None else n_samples
        future_df = self.make_future_dataframe(horizon)

        if t_scaled is None:
            future_t_scaled = np.array(
                (pd.to_datetime(future_df['ds']) - self.ds.min())
                / (self.ds.max() - self.ds.min()))
        else:
            future_t_scaled = np.asarray(t_scaled, dtype=float)

        if self.growth == 'logistic' and cap_scaled is None:
            raise ValueError(
                'Capacities must be supplied for logistic growth in column "cap"')

        # [fc] the rate of the Poisson process is S per unit of scaled time,
        # so a frame reaching T sees S * (T - 1) new changepoints on average.
        horizon_scaled = float(future_t_scaled.max())
        n_changepoints = len(self.t_change)
        # [fc] `+ 1e-8`: a fit with no active changepoints gives mean|delta| = 0,
        # and Laplace(0, 0) is undefined. Without it a perfectly straight series
        # produced a zero-width band rather than a narrow one.
        lambda_mle = float(np.abs(delta).mean()) + 1e-8

        forecast = self._sample_trends(future_t_scaled, cap_scaled, floor, n_samples)
        quantiles = self._quantiles(forecast)

        return future_df, quantiles
    
    def predict(self, future_df):
        # A copy up front: `t_scaled` is added below, and writing a column into
        # the caller's frame is theirs to be surprised by (#35).
        future_df = self._ensure_regressor_values(future_df).copy()

        # Extract optimal parameters
        k, m, delta, _sigma_obs, beta = extract_params(self.opt_params, self.layout)
        
        # Normalize future dates
        future_df['t_scaled'] = (pd.to_datetime(future_df['ds']) - self.ds.min()) / (self.ds.max() - self.ds.min())
        
        # Trend component calculation. Logistic growth needs the future frame's
        # own capacities -- they are data, and may well differ from the
        # history's. [fc] predict re-runs setup_dataframe for the same reason.
        cap_scaled, floor = self._future_capacity(future_df)
        trend = predict_trend(k, m, delta, self.t_change,
                              future_df['t_scaled'].values, self.y_scale,
                              cap_scaled, floor, self.growth)

        # Seasonality, the regressor draws and the yhat draws, all from one
        # pass: yhat's interval is the spread of its own draws rather than the
        # trend's band shifted, which is what lets the observation noise (#63)
        # and a regressor's own forecast uncertainty (#16 task 14a) enter it.
        _, trend_draws, (multiplier, seasonality), yhat_draws = self._forecast_draws(
            future_df, cap_scaled, floor, self.uncertainty_samples)

        # [stan] trend .* (1 + X_sm * beta) + X_sa * beta. `trend` is already in
        # the series' own units, and the multiplier is unitless, so only the
        # additive part needs de-normalizing.
        yhat = trend * multiplier + seasonality * self.y_scale

        forecast = future_df[['ds']].copy()
        forecast['trend'] = trend

        trend_quantiles = self._quantiles(trend_draws)
        forecast['trend_lower'] = trend_quantiles[0, :]
        forecast['trend_upper'] = trend_quantiles[1, :]

        yhat_quantiles = self._quantiles(yhat_draws)
        forecast['yhat_lower'] = yhat_quantiles[0, :]
        forecast['yhat_upper'] = yhat_quantiles[1, :]

        # the additive part in the series' units, plus what the multiplicative
        # part contributes at the fitted trend -- together, yhat - trend
        forecast['seasonality'] = seasonality * self.y_scale + trend * (multiplier - 1.0)

        forecast['yhat'] = yhat

        return forecast