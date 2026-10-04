import copy
import glob
import importlib
import importlib.util
import collections
import logging
import os
import numpy as np
import pandas as pd
from scipy.optimize import OptimizeResult, approx_fprime, minimize
from scipy.stats import halfcauchy
from typing import Tuple

from .constants import (  # noqa: F401 -- re-exported, and the
    # names tests monkeypatch must stay bound in this module
    N_CHANGE_POINTS, CHANGEPOINT_RANGE, n_yearly, TAU, SIGMA, SIGMA_OBS_PRIOR_SCALE, sigma_k, sigma_m, SIGMA_OBS_INIT)
from .layout import (  # noqa: F401 -- re-exported, and the
    # names tests monkeypatch must stay bound in this module
    ParameterLayout, extract_params, from_dict_to_array, canonical_to_split, split_to_canonical, DEFAULT_LAYOUT, K_IDX, M_IDX, DELTA_SLICE, SIGMA_OBS_IDX, BETA_SLICE)
from .seasonality import (  # noqa: F401 -- re-exported, and the
    # names tests monkeypatch must stay bound in this module
    seasonality, history_spacing, parse_seasonality_args, check_seasonality_supported, condition_masks, make_seasonality_features, condition_matrix, seasonality_columns, seasonality_modes, seasonality_prior_scales, seasonal_time, fourier_series, SEASONALITY_DEFAULTS, HONORED_SEASONALITY_FIELDS, BUILT_IN_NAMES, _RESERVED_STEMS, RESERVED_COLUMN_NAMES, BUILT_IN_SEASONALITIES, AUTO_SEASONALITY_RULES, UNDER_IDENTIFIED_WARNING, FOURIER_EPOCH, YEARLY_PERIOD)
from .make_holidays import (  # noqa: F401 -- re-exported, and the
    # names tests monkeypatch must stay bound in this module
    _country_holidays_class, get_holiday_names, make_holidays_df, validate_holidays_frame, make_holiday_features, COUNTRY_CODE_SUBSTITUTIONS, COUNTRY_NAME_SWEEP, HOLIDAYS_PACKAGE_HINT)
from .trend import (  # noqa: F401 -- re-exported, and the
    # names tests monkeypatch must stay bound in this module
    flat_growth_init, logistic_gamma_and_jacobian, logistic_trend_and_jacobian, logistic_growth_init, linear_growth_init, det_dot, predict_trend, TREND_INDICATORS)
from .optimizer import (  # noqa: F401 -- re-exported, and the
    # names tests monkeypatch must stay bound in this module
    finite_difference_hessian, projected_newton, STAN_EPS, STAN_TOL_OBJ, STAN_TOL_REL_OBJ, STAN_TOL_GRAD, STAN_TOL_PARAM, STAN_MAX_ITERATIONS, SCIPY_TOL_REL_OBJ, NEWTON_BELOW, CONSTANT_SERIES_SIGMA_OBS, SCIPY_LINE_SEARCH_FAILURE, CPP_SOLVER_RAISED)
from .models import (  # noqa: F401 -- re-exported, and the
    # names tests monkeypatch must stay bound in this module
    load_cpp_module, canonical_to_cpp, cpp_to_canonical, CPP_MODULE_NAME, BUILD_HINT, _cpp_module_cache)

logger = logging.getLogger("analytic_prophet")

# How many trend draws share one batched evaluation. The temporaries are
# (chunk, T, S + new changepoints) doubles, so this trades memory against the
# Python-level loop the draws used to be -- 64 keeps them in the tens of
# megabytes on the longest series here (#87).
TREND_DRAW_CHUNK = 64

# Every attribute a fit derives from the data, as opposed to the configuration
# the user set. `_reset_fit_state` restores each of these to the value the
# constructor gave it before a fit runs, which is what makes a refit equivalent
# to a fresh instance with the same configuration (#41).
#
# [fc] Prophet has no equivalent, because `Prophet.fit` raises on a second call:
#
#     if self.history is not None:
#         raise Exception('Prophet object can only be fit once. '
#                         'Instantiate a new object.')
#
# This implementation allows the second call, so it needs the contract that
# refusal stands in for. The names are listed rather than derived, and
# tests/test_refit_contract.py is what keeps the list honest -- it fits a fresh
# instance and a refit on the same data across a matrix of configurations and
# compares every attribute, so a stateful feature that forgets to reset shows up
# as a failure rather than as a wrong number.
#
# Not listed, because they are handled where they are set:
#   seasonalities        set_auto_seasonalities clears what it previously
#                        registered (self._auto_registered) and leaves
#                        hand-registered components alone
#   n_changepoints       set_changepoints caps the *configured* count each time
#   extra_regressors     the fitted mu/std are overwritten on every fit
FIT_DERIVED_ATTRIBUTES = (
    # the history itself
    "y", "ds", "t", "T", "t_seasonality", "y_scaled", "y_scale",
    "cap_scaled", "floor",
    # generated changepoints -- restoring the constructed value keeps a
    # user-supplied list and clears a generated one, which is the right answer
    # for both
    "changepoints", "changepoints_t",
    # the design matrix and everything describing its columns
    "layout", "sigmas", "s_a", "s_m", "_multiplicative", "condition_masks",
    "_fit_design_matrix", "_data_columns", "_data_prior_scales", "_data_modes",
    "_data_column_count", "_holiday_columns", "_holiday_prior_scales",
    "train_holiday_names", "train_holiday_column_names",
    "_regressor_history",
    # the result
    "params", "_params_vector", "k", "m", "delta", "beta", "sigma_obs",
    "opt", "loss_over_iterations", "_fitted_with_cpp", "_fit_lib_path",
    "predicted_vectorized",
)

# Set only by a fit, so there is no constructed value to go back to. A refit
# through the other path would otherwise read a status its own run never wrote.
FIT_ONLY_ATTRIBUTES = ("opt_status", "opt_status_message", "optimizer_used")

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

class AnalyticProphet:

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
        every derivation in this repository call `changepoint_prior_scale`. The argument takes
        Prophet's name and the attribute keeps Stan's.
        """
        self.rng = np.random.default_rng()

        self.t = None
        self.y = None
        self.y_scaled = None
        self.y_scale = None
        self.ds = None

        self.T = None
        self.growth = growth
        self.cap_scaled = None
        self.floor = None
        # [fc] an explicit list sets n_changepoints from its length and marks
        # the model as specified; otherwise the count stands and the dates are
        # generated at fit time. The flag is what keeps a refit from reading
        # the previous fit's generated dates back as a user-supplied list.
        if changepoints is not None:
            self.changepoints = pd.Series(pd.to_datetime(changepoints), name="ds")
            self.n_changepoints = len(self.changepoints)
            self.specified_changepoints = True
        else:
            self.changepoints = changepoints
            self.n_changepoints = n_changepoints
            self.specified_changepoints = False
        # set_changepoints fills these in when it caps `n_changepoints`; see
        # _reset_fit_state for why both halves are needed.
        self._n_changepoints_before_cap = None
        self._n_changepoints_after_cap = None
        self.changepoints_t = None
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
        self.train_holiday_column_names = None
        self._fit_design_matrix = None
        # [fc] Prophet.logistic_floor: whether the *history* carried a `floor`
        # column, which is what makes one mandatory on every later frame.
        self.logistic_floor = False
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
        self.changepoint_prior_scale = changepoint_prior_scale
        self.sigmas = seasonality_prior_scales({})
        self.s_m = np.empty(0)
        self.s_a = np.empty(0)
        self._multiplicative = False
        self.sigma_obs = SIGMA_OBS_INIT

        self.interval_width = interval_width
        self.uncertainty_samples = uncertainty_samples
        # [fc] IStanBackend.__init__ sets newton_fallback = True. One retry with
        # Newton when L-BFGS fails, at any series length.
        self.newton_fallback = True

        self.m = None
        self.k = None
        self.delta = None
        self.beta = None

        # [fc] Prophet.params: a dict of arrays, each with a leading sample
        # axis of length 1 for a MAP fit. The optimizer works in a flat vector,
        # which stays private -- `params` is the fitted result, not the thing
        # being optimized.
        self.params = None
        self._params_vector = None
        self.loss_over_iterations = None
        self.opt = None
        # which uncertainty sampler the last predict used (#93)
        self.predicted_vectorized = None

        self.sigma_k = sigma_k
        self.sigma_m = sigma_m

        self.t_seasonality = None

        self._reject_unsupported(mcmc_samples, stan_backend, scaling)
        # Taken before add_holidays, which is configuration rather than fit
        # state -- it sets `self.holidays`, which is not in the list.
        self._constructed_fit_state = {
            name: copy.deepcopy(getattr(self, name))
            for name in FIT_DERIVED_ATTRIBUTES}
        if holidays is not None:
            self.add_holidays(holidays)

    @staticmethod
    def _reject_unsupported(mcmc_samples, stan_backend, scaling):
        """Refuse what Prophet accepts and this does not.

        Accepting these silently would be the failure mode the whole of #16
        exists to avoid, and omitting them would fail with an AttributeError
        that says nothing about why.
        """
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
        """The fitted parameters as the flat canonical vector.

        No Prophet counterpart -- `params` is the dict both carry, and this is
        the vector the optimizer worked in. It reads fitted state, so it
        answers the same way as everything else that does (#104).
        """
        if self._params_vector is None:
            raise ValueError('Model has not been fit.')
        return self._params_vector

    def _needs_newton_fallback(self, failed, params) -> bool:
        """Whether an L-BFGS run should be retried with Newton.

        [fc] the retry is triggered by a `RuntimeError` out of cmdstanpy, which
        is what Stan raises when the optimizer exits non-zero. Nothing raises on
        either path here -- both solvers report a status instead -- so each
        caller translates its own status into `failed` (see
        SCIPY_LINE_SEARCH_FAILURE and CPP_SOLVER_RAISED) and a non-finite result
        counts regardless of what the status says, which is #13's failure mode.
        """
        if not self.newton_fallback:
            return False
        return failed or not np.all(np.isfinite(params))

    def _store_params(self, vector):
        """Unpack the optimizer's flat vector into Prophet's `params` dict.

        [fc] the shapes carry a leading axis of length 1, which is the MCMC
        sample dimension -- one row for a MAP fit. Keeping it means
        `params['k'][0]` reads the same here as there.
        """
        k, m, delta, sigma_obs, beta = extract_params(vector, self.layout)
        self._params_vector = vector
        self.params = {
            "k": np.array([[k]]),
            "m": np.array([[m]]),
            "delta": np.asarray(delta, dtype=float).reshape(1, -1),
            "sigma_obs": np.array([[sigma_obs]]),
            "beta": np.asarray(beta, dtype=float).reshape(1, -1),
        }
        # [fc] the fitted trend over the history, which Prophet carries here too
        self.params["trend"] = predict_trend(
            k, m, delta, self.changepoints_t, self.t, self.y_scale,
            self.cap_scaled, self.floor, self.growth).reshape(1, -1)
        # [fc] Prophet carries Stan's log posterior here; ours is the negative
        # of the objective the optimizer minimised.
        self.params["lp__"] = np.array([[-self._minus_log_posterior(vector)]])

    def _reset_fit_state(self):
        """Forget everything the previous fit derived, keeping the configuration.

        [fc] nothing -- `Prophet.fit` raises on a second call rather than
        defining what one would mean. This implementation allows refits, so it
        owes a contract instead, and this is it: **a refit is equivalent to a
        fresh instance carrying the same user configuration, fit on the new
        data.** What the user set survives; what the last history produced does
        not.

        It is not hypothetical tidiness. Without it, a model fit on twenty rows
        kept `n_changepoints` capped at 15 and fitted 15 rather than 25 on the
        next history, and a model fit across one date range kept that range's
        holiday names and forced them as all-zero columns onto the next.
        Neither raised; both silently fitted a different model (#41).
        """
        for name, value in self._constructed_fit_state.items():
            setattr(self, name, copy.deepcopy(value))
        for name in FIT_ONLY_ATTRIBUTES:
            self.__dict__.pop(name, None)

        # `n_changepoints` is configuration that set_changepoints overwrites, so
        # it is neither purely one nor the other. Undoing the cap only while the
        # capped value is still standing is what tells the two apart: a user who
        # assigned `model.n_changepoints` between fits has theirs kept, and a
        # user who did not gets the count they configured back.
        if (self._n_changepoints_after_cap is not None
                and self.n_changepoints == self._n_changepoints_after_cap):
            self.n_changepoints = self._n_changepoints_before_cap
        self._n_changepoints_before_cap = None
        self._n_changepoints_after_cap = None

    def _fitted(self):
        """(k, m, delta, sigma_obs, beta) from `params`, in the shapes the
        arithmetic wants rather than the ones the dict carries.

        [fc] the check and its wording. Every path that reads a fitted
        parameter comes through here, so this is the one place it has to be --
        without it, predicting before fitting raised `TypeError: 'NoneType'
        object is not subscriptable` from the subscript on the line below (#104).
        """
        if self.params is None:
            raise ValueError('Model has not been fit.')
        return (float(self.params["k"][0][0]), float(self.params["m"][0][0]),
                self.params["delta"][0], float(self.params["sigma_obs"][0][0]),
                self.params["beta"][0])
        
    def _normalize_y(self) -> None:
        """y / y_scale, with Prophet's guard for a scale of zero.

        [fc] initialize_scales under `scaling='absmax'`, which is the only
        scaling here, down to `if self.y_scale == 0: self.y_scale = 1.0`.

        That guard is not a corner case worth skipping: an all-zero history
        gives y_scale = 0, and without it every y_scaled was 0/0 = nan, which
        fitted to nan parameters and forecast nan, raising nothing (#102). The
        series that triggers it is a real one -- a count of something that
        never happened over the window being fitted.
        """
        self.y_scale = np.max(np.abs(self.y))
        if self.y_scale == 0:
            self.y_scale = 1.0
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
        # [fc] setup_dataframe's `if self.logistic_floor: if 'floor' not in df:
        # raise`, wording included. A history fitted with a floor is fitted on
        # `y - floor`, so a future frame that omits it is not defaulting to
        # zero, it is asking for a different model: dropping the column from
        # the frame moved the last yhat of a 150-point logistic fit from 6.92
        # to 1.90, with nothing raised and nothing logged.
        if self.logistic_floor and 'floor' not in future_df:
            raise ValueError('Expected column "floor".')
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
        self.logistic_floor = 'floor' in df
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
        if self.params is not None:
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
        if self.params is not None:
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

    def make_all_seasonality_features(self, df):
        """The whole design matrix, its prior scales and its modes.

        [fc] Prophet.make_all_seasonality_features, which returns the same
        first two and a `component_cols` frame this implementation has no use
        for -- it exists there for plotting and the component decomposition,
        neither of which is implemented here.

        Column order is Prophet's: every seasonal component in registry order,
        then holidays, then extra regressors. That order is what `beta`,
        `sigmas` and `s_m` are all indexed by.
        """
        dates = df['ds']
        masks = condition_masks(self.seasonalities, df)

        blocks, prior_scales, modes = [], [], []
        for name, props in self.seasonalities.items():
            block = make_seasonality_features(dates, props["period"],
                                              props["fourier_order"], name)
            if name in masks:
                # [fc] features[~df[condition_name]] = 0
                block = pd.DataFrame(
                    np.where(np.asarray(masks[name], dtype=bool)[:, None],
                             block.to_numpy(dtype=float), 0.0),
                    columns=block.columns)
            blocks.append(block)
            prior_scales.extend([props["prior_scale"]] * block.shape[1])
            modes.extend([props["mode"]] * block.shape[1])

        data_columns, data_scales, data_modes = self._data_design(dates, df)
        if data_columns is not None and data_columns.shape[1]:
            names = self._data_column_names()
            blocks.append(pd.DataFrame(data_columns, columns=names))
            prior_scales.extend(data_scales)
            modes.extend(data_modes)

        if not blocks:
            return pd.DataFrame(index=range(len(df))), [], []
        features = pd.concat([b.reset_index(drop=True) for b in blocks], axis=1)
        return features, prior_scales, modes

    def _data_column_names(self):
        """Names for the holiday and regressor columns, [fc]'s own."""
        names = list(self.train_holiday_column_names or [])
        return names + list(self.extra_regressors)

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
        holidays_array = (np.empty((len(df), 0)) if holidays_block is None
                          else np.asarray(holidays_block, dtype=float))

        holiday_mode = self.holidays_mode or self.seasonality_mode
        modes = [holiday_mode] * holidays_array.shape[1]
        modes += [props["mode"] for props in self.extra_regressors.values()]

        scales = list(holiday_scales) + [props["prior_scale"]
                                         for props in self.extra_regressors.values()]
        columns = np.concatenate([holidays_array,
                                  regressor_columns(self.extra_regressors, df)], axis=1)
        return columns, scales, modes

    def _holiday_design(self, dates):
        """(features, prior_scales) for `dates`, as an array.

        make_holiday_features returns a named frame, matching Prophet's; the
        arithmetic downstream wants the values, so the conversion happens here
        rather than at every use.
        """
        frame = self.construct_holiday_dataframe(dates)
        if frame is None:
            return None, []
        features, scales, _ = make_holiday_features(dates, frame, self.holidays_prior_scale)
        return features.to_numpy(dtype=float), scales

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
        if self.params is not None:
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
        # from the changepoints themselves: set_changepoints caps the count on
        # short series ([fc]), so `n_changepoints` is a request and
        # `len(changepoints_t)` is what was placed
        n_changepoints = (self.n_changepoints if self.changepoints_t is None
                          else len(self.changepoints_t))
        self.layout = ParameterLayout(n_changepoints,
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

    def set_changepoints(self) -> None:
        """Place the potential changepoints. [fc] Prophet.set_changepoints.

        Three behaviours, all of them ported together because splitting the
        function leaves it half-right:

        **Placement is by row index, not by time.** Prophet spaces
        `n_changepoints + 1` indexes evenly across the first
        `changepoint_range` of the *rows* and takes the dates there. This
        implementation spaced them evenly in scaled *time* instead. The two
        agree on regularly-spaced data and diverge on anything gappy, since
        index-spacing follows observation density and time-spacing does not.

        **The count is capped.** `n_changepoints + 1 > floor(T * range)` caps it
        at `floor(T * range) - 1`, which bites below about 32 observations --
        15 changepoints at T = 20 where this used 25, ten more rate parameters
        than Prophet fits on twenty points.

        **An explicit list is honoured**, and must fall inside the training
        data. `changepoints=` was rejected by the constructor until now (#52).

        One consequence worth naming: Prophet's changepoints land *on actual
        observation times*, so `t[i] >= t_change[j]` is exactly on the boundary
        for one row per changepoint. With time-spaced changepoints that never
        happened here, which is why `>=` against `>` was numerically inert; it
        is not inert any more.
        """
        if self.specified_changepoints:
            if len(self.changepoints) > 0:
                changepoints = pd.to_datetime(pd.Series(np.asarray(self.changepoints)))
                if changepoints.min() < self.ds.min() or changepoints.max() > self.ds.max():
                    raise ValueError("Changepoints must fall within training data.")
                self.changepoints = changepoints
        else:
            hist_size = int(np.floor(self.T * self.changepoint_range))
            if self.n_changepoints + 1 > hist_size:
                # [fc] Prophet overwrites n_changepoints with the capped value,
                # so a user reading it after a fit sees the cap. Prophet can do
                # that unconditionally because it refuses a second fit; here the
                # count before the cap is kept, and _reset_fit_state puts it
                # back before the next fit. Without that a model fit once on
                # twenty rows would fit 15 changepoints on every later history,
                # however long (#41).
                self._n_changepoints_before_cap = self.n_changepoints
                self.n_changepoints = hist_size - 1
                self._n_changepoints_after_cap = self.n_changepoints
                logger.info("n_changepoints greater than number of observations. "
                            "Using %d.", self.n_changepoints)
            if self.n_changepoints > 0:
                indexes = np.linspace(0, hist_size - 1,
                                      self.n_changepoints + 1).round().astype(int)
                # tail(-1): the first index is the series start, which is never
                # itself a changepoint
                self.changepoints = pd.Series(np.asarray(self.ds)[indexes][1:])
            else:
                self.changepoints = pd.Series(pd.to_datetime([]), name="ds")

        if len(self.changepoints) > 0:
            scale = self.ds.max() - self.ds.min()
            self.changepoints_t = np.sort(np.asarray(
                (pd.to_datetime(self.changepoints) - self.ds.min()) / scale, dtype=float))
        else:
            # [fc] a dummy, so the design matrix keeps a column and `delta`
            # keeps an entry rather than the layout collapsing
            self.changepoints_t = np.array([0.0])

    def _design_matrices(self):
        """The changepoint indicator A (T x S) and the Fourier design matrix
        (T x K).

        Both depend only on t and changepoints_t, so they are constant
        for a whole fit -- yet rebuilding them was 57% of every objective
        evaluation (issue #28). fit() computes them once and threads them
        through; callers that pass nothing still get correct results, which is
        what keeps the objective usable on its own.
        """
        A = (self.t[:, None] >= self.changepoints_t) * 1
        # [fc] make_all_seasonality_features appends holiday columns after the
        # seasonal ones; `beta` and `sigmas` follow that order.
        # built once by the fit (#28); rebuilt here only for a model that has
        # been prepared but not fitted, which is what the test fixtures use
        x = self._fit_design_matrix
        if x is None:
            # A model prepared but not fitted, which is what the test fixtures
            # use. Condition columns are reconstructed from the stored masks,
            # since make_all_seasonality_features reads them off the frame.
            frame = pd.DataFrame({"ds": np.asarray(self.ds)})
            for name, mask in self.condition_masks.items():
                frame[self.seasonalities[name]["condition_name"]] = np.asarray(mask)
            x = np.ascontiguousarray(
                self.make_all_seasonality_features(frame)[0].to_numpy(dtype=float))
        return A, x

    def _minus_log_posterior(self, params: np.array, include_l1_prior: bool=True, design=None) -> float:
        k, m, delta, sigma_obs, beta = extract_params(params, self.layout)

        A, x = design if design is not None else self._design_matrices()

        # trend component. Linear keeps its own expression rather than going
        # through the Jacobian form: it is the common case, and its fits are
        # sensitive to the last bits (see the note on the additive branch).
        if self.growth == 'logistic':
            g, trend_jacobian = logistic_trend_and_jacobian(
                k, m, delta, self.t, self.cap_scaled, A, self.changepoints_t)
        elif self.growth == 'flat':
            # [stan] flat_trend: rep_vector(m, T). k and delta remain
            # parameters and keep their priors, but the likelihood does not see
            # them, so both are driven to zero.
            trend_jacobian = None
            g = np.full(self.T, m)
        else:
            trend_jacobian = None
            gamma = -self.changepoints_t * delta
            g = (k + np.dot(A, delta)) * self.t + (m + np.dot(A, gamma))

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
            minus_log_posterior += np.sum(np.abs(delta)) / self.changepoint_prior_scale

        return minus_log_posterior

    def _gradient(self, params: np.array, include_l1_prior: bool=True, design=None) -> np.array:
        k, m, delta, sigma_obs, beta = extract_params(params, self.layout)

        A, x = design if design is not None else self._design_matrices()

        # trend component. Linear keeps its own expression rather than going
        # through the Jacobian form: it is the common case, and its fits are
        # sensitive to the last bits (see the note on the additive branch).
        if self.growth == 'logistic':
            g, trend_jacobian = logistic_trend_and_jacobian(
                k, m, delta, self.t, self.cap_scaled, A, self.changepoints_t)
        elif self.growth == 'flat':
            # [stan] flat_trend: rep_vector(m, T). k and delta remain
            # parameters and keep their priors, but the likelihood does not see
            # them, so both are driven to zero.
            trend_jacobian = None
            g = np.full(self.T, m)
        else:
            trend_jacobian = None
            gamma = -self.changepoints_t * delta
            g = (k + np.dot(A, delta)) * self.t + (m + np.dot(A, gamma))

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
            dk = np.array([-np.sum(r_scaled * self.t) / sigma_obs**2 + k / self.sigma_k**2])
            dm = np.array([-np.sum(r_scaled) / sigma_obs**2 + m / self.sigma_m**2])
            ddelta = -np.sum(r_scaled[:, None] * (self.t[:, None] - self.changepoints_t) * A, axis=0) / sigma_obs**2
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
            ddelta = ddelta + np.sign(delta) / self.changepoint_prior_scale

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
                k, m, delta, self.t, self.cap_scaled, A, self.changepoints_t)
        elif self.growth == 'flat':
            # [stan] flat_trend: rep_vector(m, T). k and delta remain
            # parameters and keep their priors, but the likelihood does not see
            # them, so both are driven to zero.
            trend_jacobian = None
            g = np.full(self.T, m)
        else:
            trend_jacobian = None
            gamma = -self.changepoints_t * delta
            g = (k + np.dot(A, delta)) * self.t + (m + np.dot(A, gamma))

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
            dk = np.array([-np.sum(r_scaled * self.t) / sigma_obs**2 + k / self.sigma_k**2])
            dm = np.array([-np.sum(r_scaled) / sigma_obs**2 + m / self.sigma_m**2])
            ddelta = -np.sum(r_scaled[:, None] * (self.t[:, None] - self.changepoints_t) * A, axis=0) / sigma_obs**2
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
            minus_log_posterior += np.sum(np.abs(delta)) / self.changepoint_prior_scale
            ddelta = ddelta + np.sign(delta) / self.changepoint_prior_scale

        gradient = np.concatenate([dk, dm, ddelta, dsigma_obs, dbeta])

        return minus_log_posterior, gradient

    # --- split-space (delta = delta_pos - delta_neg) wrappers --------------
    # These are what fit() optimizes over. They evaluate the posterior without
    # its L1 term and add (delta_pos + delta_neg)/changepoint_prior_scale instead, which is the
    # same value but differentiable -- see canonical_to_split for why.

    def _split_l1_penalty(self, z):
        n_delta = self.layout.n_changepoints
        return np.sum(z[2:2 + 2 * n_delta]) / self.changepoint_prior_scale

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
        """d/d(delta_pos) = d/d(delta) + 1/changepoint_prior_scale, d/d(delta_neg) = -d/d(delta) + 1/changepoint_prior_scale."""
        ddelta = gradient[self.layout.delta]
        return np.concatenate((
            gradient[:2],
            ddelta + 1 / self.changepoint_prior_scale,
            -ddelta + 1 / self.changepoint_prior_scale,
            [gradient[self.layout.sigma_obs_idx]],
            gradient[self.layout.beta],
        ))

    def _clean_history(self, df: pd.DataFrame) -> pd.DataFrame:
        """The history a fit can actually use, or a sentence saying why not.

        [fc] Prophet.preprocess and setup_dataframe between them, including the
        wording: someone porting a script should recognise the error and be able
        to search for it. Prophet's rule is three-way and is copied rather than
        approximated -- a missing `y` is **dropped**, a missing `ds` **raises**,
        and an infinite `y` **raises**.

        Dropping is right for `y` because the trend is a function of `t`: a
        dropped row leaves a real gap in the design matrix rather than shifting
        everything after it up by one. It is wrong for `ds`, where there is no
        timestamp to leave a gap at.

        Before #102 none of this existed and a single NaN in `y` fitted happily
        to NaN, returning an all-NaN forecast with nothing raised and nothing
        logged -- the failure mode of #13, on the input side.
        """
        if ('ds' not in df) or ('y' not in df):
            raise ValueError(
                'Dataframe must have columns "ds" and "y" with the dates and '
                'values respectively.')

        history = df[df['y'].notnull()].copy()
        if history.shape[0] < 2:
            raise ValueError('Dataframe has less than 2 non-NaN rows.')
        dropped = len(df) - len(history)
        if dropped:
            # Prophet drops them silently. This does not, which is the one
            # place here that says more than Prophet does: a fit on fewer rows
            # than the caller passed is the event whose silence made #102 hard
            # to notice at all. INFO, so a caller who is not listening sees no
            # change.
            logger.info("Dropping %d row(s) with a missing y.", dropped)

        history['y'] = pd.to_numeric(history['y'])
        if np.isinf(history['y'].values).any():
            raise ValueError('Found infinity in column y.')

        dates = pd.to_datetime(history['ds'])
        # the timezone check comes first, [fc] setup_dataframe's order. It only
        # shows on a frame that is both tz-aware and holds a NaT, where the two
        # messages compete -- and there Prophet names the timezone.
        if dates.dt.tz is not None:
            raise ValueError('Column ds has timezone specified, which is not '
                             'supported. Remove timezone.')
        if dates.isnull().any():
            raise ValueError('Found NaN in column ds.')
        history['ds'] = dates
        return history.reset_index(drop=True)

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Everything both fit paths do to the history before optimizing it.

        [fc] Prophet.preprocess, and at the same seam: it reformats the history,
        normalizes y, fits the regressor standardizations, selects the
        seasonalities, places the changepoints and builds the design matrix.
        Prophet's also returns a `ModelInputData`; this one only saves to
        `self`, because both callers read the attributes rather than a record.

        It exists because `fit` and `fit_cpp` each carried this block verbatim
        -- 45 identical lines, differing in one comment. Every change to
        preprocessing had to be made twice, and #41 is what that costs: both
        paths needed `_reset_fit_state()` added separately, and a miss would
        have been silent in whichever one was forgotten.

        Returns the **cleaned** history, which is not always the frame it was
        given: rows with a missing `y` are dropped (#102), and everything
        downstream -- the design matrix, the regressor standardizations, the
        nested regressor models -- has to see the same rows the optimizer will.
        """
        self._reset_fit_state()
        df = self._clean_history(df)
        self.y = df['y'].values

        if df['ds'].dtype != 'datetime64[ns]':
            self.ds = pd.to_datetime(df['ds'])
        else:
            self.ds = df['ds']

        self.t = np.array((self.ds - self.ds.min()) / (self.ds.max() - self.ds.min()))
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
        if self._holiday_columns:
            self.train_holiday_column_names = list(make_holiday_features(
                self.ds, self.construct_holiday_dataframe(self.ds),
                self.holidays_prior_scale)[0].columns)
        self.set_changepoints()
        self._build_layout()
        # built once here, not per objective evaluation (#28)
        self._fit_design_matrix = np.ascontiguousarray(
            self.make_all_seasonality_features(df)[0].to_numpy(dtype=float))
        return df

    def calculate_initial_params(self, initial_params: dict=None) -> dict:
        """The point both optimizers start from.

        [fc] Prophet.calculate_initial_params, which takes `K` and returns a
        `ModelParams`; this takes the caller's overrides and returns a dict in
        the canonical (k, m, delta, sigma_obs, beta) order. k and m come from
        the growth mode's initializer, delta and beta are zero, sigma_obs is 1.

        Prophet passes these to Stan explicitly, so Stan's random init is never
        reached. `fit_cpp` used to draw `init_r * N(0, 1)` here instead,
        described as "STAN initialization" -- from an unseeded generator, which
        made it non-reproducible run to run.
        """
        if self.growth == 'logistic':
            k_init, m_init = logistic_growth_init(self.t, self.y_scaled,
                                                  self.cap_scaled)
        elif self.growth == 'flat':
            k_init, m_init = flat_growth_init(self.y_scaled)
        else:
            k_init, m_init = linear_growth_init(self.t, self.y_scaled)
        params = {
            'k': k_init,
            'm': m_init,
            'delta': np.zeros(self.layout.n_changepoints),
            'sigma_obs': SIGMA_OBS_INIT,
            'beta': np.zeros(self.layout.n_regressor_columns),
        }
        if initial_params is not None:
            params.update(initial_params)
        return params

    def _fit_constant_series(self, initial_params: dict,
                             sigma_obs: float=None) -> bool:
        """Prophet's short-circuit for a series that never moves.

        [fc] Prophet.fit, condition and number both: when `y.min() ==
        y.max()` under linear or flat growth the initial parameters already
        are the answer -- a flat line through a flat series -- so it keeps
        them, sets sigma_obs to CONSTANT_SERIES_SIGMA_OBS, and never calls
        Stan at all. Returns whether it applied, so the caller can skip its
        optimizer. Logistic growth is excluded there and here: its initializer
        solves for a curve approaching `cap`, which a constant series is not.

        Optimizing instead reaches the same line and the same forecast -- both
        paths put yhat on the constant exactly -- and differs only in
        sigma_obs, where it runs down to fit()'s 1e-6 bound rather than
        Prophet's 1e-9. The likelihood has no interior optimum here, since the
        residuals are identically zero and -T*log(sigma) rises without limit
        as sigma falls, so the estimate is whichever floor it is given and the
        intervals are degenerate either way (width 7.7e-6 against Prophet's
        7.7e-9 on a 200-point constant series). What matching buys is the
        iterations, and the parity (#102).
        """
        if self.growth not in ('linear', 'flat'):
            return False
        if self.y.min() != self.y.max():
            return False

        params = dict(initial_params)
        params['sigma_obs'] = (CONSTANT_SERIES_SIGMA_OBS if sigma_obs is None
                               else sigma_obs)
        vector = from_dict_to_array(params, self.layout)
        self.opt = None
        # no optimizer ran, and saying so is better than naming one that did not
        self.optimizer_used = None
        self.loss_over_iterations = []
        self.sigma_obs = vector[self.layout.sigma_obs_idx]
        self._store_params(vector)
        return True

    def fit(self, df: pd.DataFrame, analytic: bool=False, use_combined: bool=False, optimizer: str='L-BFGS-B',
            initial_params: dict=None, fixed_sigma_obs: float=None,
            algorithm: str=None) -> Tuple[float, float, np.array, np.array]:
        if analytic and use_combined:
            raise ValueError("Both 'analytic' and 'use_combined' cannot be True at the same time.")

        # the cleaned history, not the caller's frame: rows with a missing y
        # are dropped, and the regressor models below must see the same rows
        # the optimizer did (#102)
        df = self.preprocess(df)

        initial_params_dict = self.calculate_initial_params(initial_params)
        if fixed_sigma_obs is not None:
            initial_params_dict['sigma_obs'] = fixed_sigma_obs

        # [fc] the constant-series branch, which Prophet also takes after the
        # initial parameters are in hand and the regressor models are fitted --
        # hence the call below, which the branch owes before it returns.
        #
        # An explicitly pinned sigma_obs is honoured over Prophet's 1e-9: it is
        # a request from the caller, which Prophet has no equivalent of, and
        # overriding it would make `fixed_sigma_obs` silently not apply.
        if self._fit_constant_series(initial_params_dict, fixed_sigma_obs):
            self._fitted_with_cpp = False
            self._fit_regressor_models(df)
            return

        loss_over_iterations = []

        # Built once here, not rebuilt on every evaluation (#28).
        design = self._design_matrices()

        def callback(z):
            # The *split* objective, which is what scipy is minimizing. It used
            # to record the canonical one, which is a different function away
            # from the optimum -- |d+ - d-| against d+ + d-, equal only when at
            # most one of each pair is non-zero. The two agree at the answer, so
            # the final value was right and the defect stayed invisible, but the
            # recorded trajectory could rise (0.031 at T=1000) where the run
            # itself descends monotonically, and it was not comparable with
            # fit_cpp's trace, which is the split objective throughout.
            loss_over_iterations.append(self._split_minus_log_posterior(z, design=design))

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
            # Stan's iteration cap and gradient tolerance apply directly. One
            # parameter does not, and #24 is the record of why.
            #
            # gtol carries Stan's tol_grad. It used to be disabled, on the
            # reading that scipy tests the inf-norm of the *projected* gradient
            # while Stan tests the 2-norm of the full one, and that on the split
            # problem the projected norm is the far smaller of the two. Whatever
            # that was worth when it was written, it is now a no-op: enabling it
            # changes neither the iteration count nor the fitted point at T =
            # 300, 1000 or 2905, or at any short size. Prophet's changepoint
            # placement (#15) is the likely reason -- index-spaced changepoints
            # land on observations, and the problem is better conditioned for it.
            #
            # ftol stays tighter than Stan's tol_rel_obj * eps = 2.22e-12, and
            # this one is real. scipy's iterate sequence on this objective has
            # single steps that barely move followed by steps that move a lot:
            # at T=2905 it takes a step with a relative decrease of 6.8e-16 --
            # four orders below Stan's threshold -- while still 4.80 nats from
            # the optimum, then resumes descending. 37% of its steps are below
            # the threshold. Stan's test is a one-step test, so it fires on the
            # first of them, and the fit lands 4.80 nats short.
            #
            # That is not an artefact of scipy's own stopping rule: the same
            # thing happens when Stan's test is evaluated by hand on the true
            # objective. It is also not the line search (maxls 20/60/100 give
            # the identical trajectory) and not the parameterization (moving
            # sigma_obs to zeta = log(sigma_obs), which is what the C++ core
            # optimizes, fixes T=300 and T=2905 and breaks T=1000 instead). The
            # C++ core runs the same split reformulation under Stan's real
            # tolerance and converges, so what differs is the iterate sequence,
            # not the problem or the test.
            #
            # The deciding measurement is against Prophet itself, which is what
            # #24 said it was blocked on. Under Stan's ftol, fit() scores
            # lp__ = 8000.371 at T=2905 against Prophet's 8004.798: matching
            # Stan's number numerically would put this path 4.43 nats *below*
            # the model it is reproducing. Tightening instead costs iterations
            # and nothing else.
            #
            # maxfun is Stan's cap expressed the other way round: Stan limits
            # iterations only, and scipy's default of 15000 evaluations would
            # otherwise end the run before maxiter does.
            options.update({
                'ftol': SCIPY_TOL_REL_OBJ,
                'gtol': STAN_TOL_GRAD,
                'maxfun': STAN_MAX_ITERATIONS * 10,
            })

        def run_newton():
            """The Newton branch, on the same split space and the same bounds.

            It takes its gradient from the same selection the L-BFGS branch
            does, so `analytic=False` stays the control it is meant to be: the
            same optimizer on the same problem with the gradient obtained the
            expensive way, rather than the flag silently ceasing to apply.
            """
            lower = np.array([-np.inf if low is None else low for low, _ in bounds])
            upper = np.array([np.inf if high is None else high for _, high in bounds])
            if use_combined:
                value = lambda z: objective(z)[0]
                gradient = lambda z: objective(z)[1]
            elif analytic:
                value, gradient = objective, jac
            else:
                value = objective
                gradient = lambda z: approx_fprime(z, objective)
            outcome = projected_newton(value, gradient, z0, lower, upper)
            loss_over_iterations[:] = outcome.loss_trace
            return outcome

        # [fc] CmdStanPyBackend.fit: Newton below 100 observations, L-BFGS at or
        # above, and one Newton retry when L-BFGS exits abnormally -- at any
        # length. An explicit `algorithm` overrides the rule, as `args.update(
        # kwargs)` does there. It is Prophet's behaviour being reproduced rather
        # than this project's preference, and measured it costs no accuracy:
        # Newton lands where L-BFGS lands at every size tested. It does cost
        # time -- 2n gradient evaluations an iteration for the Hessian. See the
        # README, "Prophet's rule for short series, and what it costs".
        #
        # `optimizer` names the scipy method behind the L-BFGS branch; naming a
        # different one is a request for that method, not for Newton.
        if algorithm is None and optimizer == 'L-BFGS-B':
            algorithm = "Newton" if self.T < NEWTON_BELOW else "LBFGS"
        self.optimizer_used = algorithm or "LBFGS"
        if self.optimizer_used == "Newton":
            result = run_newton()
        else:
            result = minimize(objective,
                            z0,
                            method=optimizer,
                            bounds=bounds,
                            options=options,
                            callback=callback,
                            jac=jac)
            if self._needs_newton_fallback(
                    result.status == SCIPY_LINE_SEARCH_FAILURE, result.x):
                logger.warning("Optimization terminated abnormally. "
                               "Falling back to Newton.")
                self.optimizer_used = "Newton"
                del loss_over_iterations[:]
                result = run_newton()

        self.opt = result
        vector = split_to_canonical(result.x, self.layout.n_changepoints)
        self.sigma_obs = vector[self.layout.sigma_obs_idx]
        self._store_params(vector)
        self._fitted_with_cpp = False
        self._fit_regressor_models(df)
        self.loss_over_iterations = loss_over_iterations
    
    def fit_cpp(self, df: pd.DataFrame, initial_params: dict=None, lib_path: str=None,
                verbose: bool=False, algorithm: str=None) -> Tuple[float, float, np.array, np.array]:
        """Fit via the compiled C++ core (see load_cpp_module for how it's found)."""
        # the cleaned history, not the caller's frame: rows with a missing y
        # are dropped, and the regressor models below must see the same rows
        # the optimizer did (#102)
        df = self.preprocess(df)

        defaults = self.calculate_initial_params(initial_params)

        # [fc] the constant-series branch, which both paths share. It comes
        # before the library is loaded, so a flat series is answered without
        # one -- the C++ core would be asked to minimise a function with no
        # interior minimum.
        if self._fit_constant_series(defaults):
            self._fitted_with_cpp = True
            self._fit_lib_path = lib_path
            self._fit_regressor_models(df)
            return -1

        # The compiled optimizer's layout is [k, m, delta(S), beta(K), zeta],
        # with zeta = log(sigma_obs) last -- see cpp_to_canonical. Estimating
        # the log keeps sigma_obs positive without box constraints, which
        # liblbfgs does not have.
        zeta_init = np.log(defaults['sigma_obs'])
        params = np.concatenate(([defaults['k']], [defaults['m']],
                                 defaults['delta'], defaults['beta'], [zeta_init]))

        cpp = load_cpp_module(lib_path)
        arguments = dict(
            params=params,
            t=self.t,
            changepoints_t=self.changepoints_t,
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
            changepoint_prior_scale=self.changepoint_prior_scale,
            fourier_orders=[props["fourier_order"] for props in self.seasonalities.values()],
            seasonality_periods=[props["period"] for props in self.seasonalities.values()],
            seasonality_conditions=condition_matrix(self.seasonalities,
                                                    self.condition_masks, self.T),
            data_columns=(self._data_columns if self._data_column_count
                              else np.empty((self.T, 0))),
            verbose=verbose,
        )

        # [fc] CmdStanPyBackend.fit: Newton below 100 observations, L-BFGS at or
        # above, and one Newton retry when L-BFGS exits abnormally -- at any
        # length. An explicit `algorithm` overrides the rule, as `args.update(
        # kwargs)` does there. It is Prophet's behaviour being reproduced rather
        # than this project's preference, and measured it costs no accuracy:
        # Newton lands where L-BFGS lands at every size tested. It does cost
        # time -- 2n gradient evaluations an iteration for the Hessian, which
        # makes a short fit about 6x slower. See the README, "Prophet's rule
        # for short series, and what it costs".
        self.optimizer_used = algorithm or ("Newton" if self.T < NEWTON_BELOW
                                            else "LBFGS")
        if self.optimizer_used == "Newton":
            result = cpp.newton(**arguments)
        else:
            result = cpp.optimize(**arguments)
            if self._needs_newton_fallback(result.status == CPP_SOLVER_RAISED,
                                            result.params):
                logger.warning("Optimization terminated abnormally. "
                               "Falling back to Newton.")
                self.optimizer_used = "Newton"
                result = cpp.newton(**arguments)

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
        # which fit method produced the parameters.
        vector = cpp_to_canonical(result.params, self.layout)
        self.sigma_obs = vector[self.layout.sigma_obs_idx]
        self._store_params(vector)
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
        if self.params is not None:
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
            AnalyticProphet(**predictor_spec)

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

            predictor = AnalyticProphet(**props["predictor_spec"])
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
        # [fc] `if self.history_dates is None: raise ... 'Model has not been
        # fit.'`. Without it this was `AttributeError: 'NoneType' object has no
        # attribute 'max'` from the line below (#104).
        if self.ds is None:
            raise ValueError('Model has not been fit.')
        last_date = pd.to_datetime(self.ds.max())  # Ensure last_date is a datetime object
        future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, periods + 1)]
        future_dates_df = pd.DataFrame(future_dates, columns=['ds'])

        if include_history:
            history_dates_df = pd.DataFrame(self.ds, columns=['ds'])
            future_df = pd.concat([history_dates_df, future_dates_df], ignore_index=True)
        else:
            future_df = future_dates_df

        return future_df
    
    def _regressor_draws(self, future_df, n_samples, vectorized=True):
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
                columns[:, is_future] = predictor._sample_yhat(
                    future_frame, n_samples, vectorized)
            drawn[name] = columns
        return drawn

    def _sample_yhat(self, future_df, n_samples, vectorized=True):
        """`n_samples` x T draws of this model's own yhat, for a nested model
        to hand its parent."""
        prepared = self._ensure_regressor_values(future_df).copy()
        prepared['t'] = ((pd.to_datetime(prepared['ds']) - self.ds.min())
                                / (self.ds.max() - self.ds.min()))
        cap_scaled, floor = self._future_capacity(prepared)
        _, _, _, yhat_draws = self._forecast_draws(prepared, cap_scaled, floor,
                                                   n_samples, vectorized)
        return yhat_draws

    def _forecast_draws(self, future_df, cap_scaled, floor, n_samples,
                        vectorized=True):
        """(x, trend_draws, seasonality, yhat_draws) for a prepared frame.

        `yhat` is drawn rather than derived from the trend band, which is what
        lets two things enter that could not before: a regressor's own forecast
        uncertainty (#16 task 14a) and the observation noise (#63), the latter
        being the term that dominates the interval.
        """
        _k, _m, _delta, sigma_obs, beta = self._fitted()
        t = future_df['t'].values

        x = np.ascontiguousarray(
            self.make_all_seasonality_features(future_df)[0].to_numpy(dtype=float))

        trend_draws = (self._sample_trends_vectorized(t, cap_scaled, floor, n_samples)
                       if vectorized and self.growth != 'logistic'
                       else self._sample_trends(t, cap_scaled, floor, n_samples))
        regressor_draws = self._regressor_draws(future_df, n_samples, vectorized)

        # the regressor columns are the last of the design matrix, in the order
        # they were registered
        positions = {name: x.shape[1] - len(self.extra_regressors) + i
                     for i, name in enumerate(self.extra_regressors)}

        # [fc] sample_model adds normal(0, sigma_obs) per draw; without it the
        # interval is the trend's alone and ~18x too narrow (#63). Drawn as one
        # (n_samples, T) block rather than a row at a time: numpy fills it in C
        # order, so the stream and the generator's end state are identical to
        # the loop's, which a test asserts.
        noise = self.rng.normal(0.0, sigma_obs * self.y_scale, (n_samples, len(t)))

        def components(design):
            multiplier = 1.0 + design.dot(self.s_m * beta) if self._multiplicative else 1.0
            seasonal = (design.dot(self.s_a * beta) if self._multiplicative
                        else design.dot(beta))
            return multiplier, seasonal

        if regressor_draws:
            # the design matrix differs per draw, so this stays a loop
            yhat_draws = np.empty_like(trend_draws)
            for i in range(n_samples):
                x_draw = x.copy()
                for name, values in regressor_draws.items():
                    props = self.extra_regressors[name]
                    x_draw[:, positions[name]] = (values[i] - props["mu"]) / props["std"]
                multiplier, seasonal = components(x_draw)
                yhat_draws[i] = trend_draws[i] * multiplier + seasonal * self.y_scale + noise[i]
        else:
            # nothing in here varies with the draw: the design matrix is fixed,
            # so the seasonal term and the multiplicative factor were being
            # recomputed identically n_samples times -- a full (T, K) product
            # each. Hoisted, the draws are one array expression, and every
            # element is the same arithmetic in the same order as the loop did
            # it (#87).
            multiplier, seasonal = components(x)
            yhat_draws = trend_draws * multiplier + seasonal * self.y_scale + noise

        multiplier = 1.0 + x.dot(self.s_m * beta) if self._multiplicative else 1.0
        seasonality = x.dot(self.s_a * beta) if self._multiplicative else x.dot(beta)
        return x, trend_draws, (multiplier, seasonality), yhat_draws

    def _quantiles(self, draws):
        """[fc] predict_uncertainty: a centred interval, so the edges are
        (1 -+ interval_width) / 2."""
        lower = 100 * (1.0 - self.interval_width) / 2
        upper = 100 * (1.0 + self.interval_width) / 2
        return np.percentile(draws, [lower, upper], axis=0)

    def _trend_chunk(self, draws, chunk, sampled, k, m, delta, t, cap_scaled, floor):
        """Fill `chunk` of `draws`, whose members all sampled the same count.

        Same count means same shape, so the whole chunk is one array
        expression. It stays **bit-identical** to evaluating each draw on its
        own because `det_dot` reduces over the last axis, and batching adds a
        leading axis without changing that axis's length or its contiguity --
        the same 25 + n_new doubles, summed the same way. Splitting the sum
        instead, into the fitted part plus the new part, would not: summing 25
        terms and then n more is a different grouping from summing 25 + n.

        Chunked so the (chunk, T, S + n_new) temporaries stay bounded; at the
        default that is tens of megabytes rather than hundreds.
        """
        changepoints = np.stack([
            np.concatenate((self.changepoints_t, sampled[i][1])) for i in chunk])
        deltas = np.stack([np.concatenate((delta, sampled[i][2])) for i in chunk])

        if self.growth == 'logistic':
            # a different function of the same pieces, and rare enough not to
            # be worth a second batched implementation
            for i in chunk:
                draws[i] = predict_trend(
                    k, m, np.concatenate((delta, sampled[i][2])),
                    np.concatenate((self.changepoints_t, sampled[i][1])),
                    t, self.y_scale, cap_scaled, floor, self.growth)
            return

        # left as bool rather than `* 1`: the products are the same floats
        # either way, and an int64 copy of a (chunk, T, S + n) indicator is
        # eight bytes an element of pure memory traffic
        indicator = t[None, :, None] >= changepoints[:, None, :]
        if self.growth == 'flat':
            trend_normalized = np.broadcast_to(m, (len(chunk), len(t)))
        else:
            gammas = -changepoints * deltas
            rates = (indicator * deltas[:, None, :]).sum(axis=-1)
            offsets = (indicator * gammas[:, None, :]).sum(axis=-1)
            trend_normalized = (k + rates) * t + (m + offsets)
        trend = trend_normalized * self.y_scale
        draws[chunk] = trend if floor is None else trend + floor

    def _trend_shift_matrix(self, mean_delta, likelihood, n_future, n_samples):
        """Random slope changes, one coin per future timestep.

        [fc] Prophet._make_trend_shift_matrix. The trapezoidal average on the
        last line is Prophet's: a change that lands between two steps is split
        across both rather than applied wholly to the later one.
        """
        occurs = self.rng.uniform(size=(n_samples, n_future)) < likelihood
        shifts = self.rng.laplace(0, mean_delta, size=(n_samples, n_future)) * occurs
        shifted = np.hstack([np.zeros((n_samples, 1)), shifts])[:, :-1]
        return (shifted + shifts) / 2

    def _sample_uncertainty(self, t, n_samples):
        """(n_samples, len(t)) of trend deviation, in normalized units.

        [fc] Prophet._sample_uncertainty, which is what `predict` runs by
        default there. It is an **approximation** of the sampler
        `_sample_trends` implements, not a faster form of it, and the two differ
        in three ways worth naming (#93):

          * the Poisson process over the horizon becomes an independent coin at
            every timestep, so at most one changepoint lands per step and the
            counts agree only as the step shrinks;
          * the trend is integrated discretely, by a double cumulative sum,
            rather than evaluated from the piecewise-linear definition;
          * rows inside the history get exactly zero, which is the same choice
            the exact sampler makes for its own reasons (#58) rather than an
            approximation.

        What it buys is the cost: O(n_samples x future rows) against
        O(n_samples x T x S), and the horizon is usually a few percent of T.
        """
        t = np.asarray(t, dtype=float)
        if t.max() <= 1.0:
            return np.zeros((n_samples, len(t)))

        future = t[t > 1.0]
        n_future = len(future)
        # one timestep, in the scaled units the changepoints live in
        single_diff = (float(np.diff(future).mean()) if n_future > 1
                       else float(np.diff(self.t).mean()))
        likelihood = len(self.changepoints_t) * single_diff
        _k, _m, delta, _sigma_obs, _beta = self._fitted()
        mean_delta = float(np.abs(delta).mean()) + 1e-8

        if self.growth == 'flat':
            # no slope to change
            uncertainty = np.zeros((n_samples, n_future))
        else:
            shifts = self._trend_shift_matrix(mean_delta, likelihood, n_future, n_samples)
            # slope changes -> slopes -> values, then scaled by what a step means
            uncertainty = shifts.cumsum(axis=1).cumsum(axis=1) * single_diff

        n_past = int(np.sum(t <= 1.0))
        if n_past:
            uncertainty = np.concatenate(
                [np.zeros((n_samples, n_past)), uncertainty], axis=1)
        return uncertainty

    def _sample_trends_vectorized(self, t, cap_scaled, floor, n_samples):
        """`n_samples` x T trend draws, Prophet's approximate way.

        [fc] sample_predictive_trend_vectorized: the fitted trend, plus a
        sampled deviation, both in normalized units and de-normalized once.
        """
        k, m, delta, _sigma_obs, _beta = self._fitted()
        expected = predict_trend(k, m, delta, self.changepoints_t, t, 1.0,
                                 cap_scaled, None, self.growth)
        uncertainty = self._sample_uncertainty(t, n_samples)
        draws = (expected[None, :] + uncertainty) * self.y_scale
        return draws if floor is None else draws + floor

    def _sample_trends(self, t, cap_scaled, floor, n_samples):
        """`n_samples` x T trend draws, in the series' own units.

        [fc] sample_predictive_trend. New changepoints come from a Poisson
        process on `(1, T]`, so they land strictly past the end of the history
        and there are none when the frame does not reach past it (#58).
        """
        k, m, delta, _sigma_obs, _beta = self._fitted()
        horizon_scaled = float(np.max(t))
        n_changepoints = len(self.changepoints_t)
        # [fc] `+ 1e-8`: a fit with no active changepoints gives mean|delta| of
        # exactly zero, and Laplace(0, 0) is degenerate -- every draw would
        # return the same trend and the band would be identically zero rather
        # than narrow.
        lambda_mle = float(np.abs(delta).mean()) + 1e-8

        # The fitted changepoints are identical in every draw, so the trend
        # they produce is evaluated once rather than a thousand times. A draw
        # that samples no new changepoints *is* that trend -- same arrays, same
        # summation -- so this is a hoist rather than an approximation (#87).
        base = predict_trend(k, m, delta, self.changepoints_t, t, self.y_scale,
                             cap_scaled, floor, self.growth)

        # All the randomness first, in the order the loop drew it, so the
        # generator ends in the same state and every draw is the same draw.
        sampled = []
        for _ in range(n_samples):
            if horizon_scaled > 1.0:
                n_new = self.rng.poisson(n_changepoints * (horizon_scaled - 1.0))
            else:
                n_new = 0
            # drawn even when n_new is 0, as the loop did
            new_change_points = np.sort(
                1.0 + self.rng.random(n_new) * (horizon_scaled - 1.0))
            new_delta = self.rng.laplace(0, lambda_mle, n_new)
            sampled.append((int(n_new), new_change_points, new_delta))

        draws = np.empty((n_samples, len(t)), dtype=base.dtype)
        remaining = collections.defaultdict(list)
        for index, (n_new, _, _) in enumerate(sampled):
            if n_new == 0:
                draws[index] = base
            else:
                remaining[n_new].append(index)

        for n_new, indices in remaining.items():
            for start in range(0, len(indices), TREND_DRAW_CHUNK):
                chunk = indices[start:start + TREND_DRAW_CHUNK]
                self._trend_chunk(draws, chunk, sampled, k, m, delta, t,
                                  cap_scaled, floor)
        return draws

    def trend_forecast_uncertainty(self, horizon=30, n_samples=None,
                                   t=None, cap_scaled=None, floor=None):
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

        `t`, `cap_scaled` and `floor` come from the caller: predict()
        passes the grid and capacities it already built, since the capacity is
        per row and the frame this method builds for itself carries only `ds`.
        """
        k, m, delta, _sigma_obs, beta = self._fitted()
        n_samples = self.uncertainty_samples if n_samples is None else n_samples
        future_df = self.make_future_dataframe(horizon)

        if t is None:
            future_t_scaled = np.array(
                (pd.to_datetime(future_df['ds']) - self.ds.min())
                / (self.ds.max() - self.ds.min()))
        else:
            future_t_scaled = np.asarray(t, dtype=float)

        if self.growth == 'logistic' and cap_scaled is None:
            raise ValueError(
                'Capacities must be supplied for logistic growth in column "cap"')

        # [fc] the rate of the Poisson process is S per unit of scaled time,
        # so a frame reaching T sees S * (T - 1) new changepoints on average.
        horizon_scaled = float(future_t_scaled.max())
        n_changepoints = len(self.changepoints_t)
        # [fc] `+ 1e-8`: a fit with no active changepoints gives mean|delta| = 0,
        # and Laplace(0, 0) is undefined. Without it a perfectly straight series
        # produced a zero-width band rather than a narrow one.
        lambda_mle = float(np.abs(delta).mean()) + 1e-8

        forecast = self._sample_trends(future_t_scaled, cap_scaled, floor, n_samples)
        quantiles = self._quantiles(forecast)

        return future_df, quantiles
    
    def predict(self, future_df, vectorized=True):
        """The forecast, and the interval around it.

        `vectorized` selects which uncertainty sampler runs, [fc] the argument
        and the default. True is Prophet's approximation -- an independent coin
        per future timestep, integrated by a double cumulative sum -- and False
        is the exact sampler, which places changepoints as a Poisson process
        and evaluates the piecewise-linear trend from its definition.

        They are **different computations, not two speeds of one**. Prophet's
        own two paths disagree by about 1.4% on the interval bounds, and so do
        these. `yhat` is unaffected: only the interval is sampled. Which one ran
        is recorded on `self.predicted_vectorized` (#93).

        The exact sampler is used regardless under logistic growth, where
        Prophet's approximation needs a separate derivation this does not have
        yet. It is slower, not wrong.
        """
        # A copy up front: `t` is added below, and writing a column into
        # the caller's frame is theirs to be surprised by (#35).
        future_df = self._ensure_regressor_values(future_df).copy()

        # Extract optimal parameters
        k, m, delta, _sigma_obs, beta = self._fitted()
        
        # Normalize future dates
        future_df['t'] = (pd.to_datetime(future_df['ds']) - self.ds.min()) / (self.ds.max() - self.ds.min())
        
        # Trend component calculation. Logistic growth needs the future frame's
        # own capacities -- they are data, and may well differ from the
        # history's. [fc] predict re-runs setup_dataframe for the same reason.
        cap_scaled, floor = self._future_capacity(future_df)
        trend = predict_trend(k, m, delta, self.changepoints_t,
                              future_df['t'].values, self.y_scale,
                              cap_scaled, floor, self.growth)

        # Seasonality, the regressor draws and the yhat draws, all from one
        # pass: yhat's interval is the spread of its own draws rather than the
        # trend's band shifted, which is what lets the observation noise (#63)
        # and a regressor's own forecast uncertainty (#16 task 14a) enter it.
        self.predicted_vectorized = bool(vectorized) and self.growth != 'logistic'
        _, trend_draws, (multiplier, seasonality), yhat_draws = self._forecast_draws(
            future_df, cap_scaled, floor, self.uncertainty_samples, vectorized)

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
