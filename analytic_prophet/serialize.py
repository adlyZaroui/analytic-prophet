"""Persist a fitted model as JSON, and load it back ready to predict.

[fc] prophet/serialize.py, down to the four entry points and the grouping of
attributes by the conversion each type needs:

    from analytic_prophet.serialize import model_to_json, model_from_json

    with open('model.json', 'w') as handle:
        handle.write(model_to_json(model))

    with open('model.json') as handle:
        model = model_from_json(handle.read())

`pickle` is deliberately not the supported route, and the reason is sharper here
than it is there. Prophet's pickles carry a Stan object and do not survive a
version change; ours would carry a handle to whatever build of `optimize.cpp`
happened to be loaded, tying the file to one machine. What is written below is
the *state a prediction reads* and nothing else -- no optimizer result, no
extension handle, no design matrix.

The attribute lists are the complement of `FIT_DERIVED_ATTRIBUTES` (#41): that
one names what a refit must forget, this one names what a load must restore.
Between them they should account for the model, and
tests/test_serialize.py::test_the_two_attribute_lists_account_for_the_model is
what keeps them reconciled.
"""
import json
from collections import OrderedDict
from copy import deepcopy
from io import StringIO

import numpy as np
import pandas as pd

from . import __version__
from .forecaster import AnalyticProphet
from .layout import ParameterLayout

# Configuration and the scalars a fit derives: JSON-native as they stand.
SIMPLE_ATTRIBUTES = [
    'growth', 'n_changepoints', 'specified_changepoints', 'changepoint_range',
    'yearly_seasonality', 'weekly_seasonality', 'daily_seasonality',
    'seasonality_mode', 'seasonality_prior_scale', 'changepoint_prior_scale',
    'holidays_prior_scale', 'holidays_mode', 'interval_width',
    'uncertainty_samples', 'country_holidays', 'sigma_k', 'sigma_m',
    'newton_fallback', 'T', 'y_scale', 'sigma_obs', '_multiplicative',
    # the cap #41 undoes. Without these two a model fitted on twenty rows,
    # saved, loaded and refitted on a longer history keeps the capped count and
    # silently fits 15 changepoints instead of 25 -- the exact bug #41 closed,
    # walked back in through the file. Prophet never faces it because its
    # models cannot be refitted at all.
    '_n_changepoints_before_cap', '_n_changepoints_after_cap',
    '_holiday_columns', '_data_column_count', '_fitted_with_cpp',
    '_holiday_prior_scales', '_data_prior_scales', '_data_modes',
    'train_holiday_names', 'train_holiday_column_names',
    # the marker a nested regressor predictor carries, [fc]'s own
    '_regressor_name',
]

# A set, so it needs a list on the way out and a set on the way back. It is
# what set_auto_seasonalities clears on a refit (#41): without it, a model
# fitted on two years of history, saved, loaded and refitted on 300 days keeps
# `yearly` -- a component the new history cannot identify, which is the bug #41
# was filed about, walked back in through the file.
SET_ATTRIBUTES = ['_auto_registered']

PD_SERIES = ['changepoints', 'ds']

PD_DATAFRAME = ['holidays', '_regressor_history']

NP_ARRAY = ['changepoints_t', 's_a', 's_m', 'sigmas', 'cap_scaled', 'floor',
            '_params_vector']

ORDEREDDICT = ['seasonalities', 'extra_regressors']

# Deliberately not written, and why:
#
#   opt, opt_status, opt_status_message, optimizer_used, loss_over_iterations
#       the optimizer's own record. [fc] Prophet skips stan_fit for the same
#       reason: a prediction never consults it.
#   _fit_lib_path, and the loaded extension behind it
#       a path to one machine's build of optimize.cpp. Writing it is how a
#       model stops loading on a machine that has never compiled the core.
#   t, y, y_scaled, t_seasonality, _fit_design_matrix, _data_columns
#       the training history in fitted form. predict() rebuilds all of it from
#       the frame it is given, and the design matrix is the largest object on
#       the model -- T x K floats, against T for `ds`.
#   condition_masks
#       computed per frame, so the training frame's masks say nothing about a
#       future one.
#   rng
#       unseeded by design; a restored model draws its own.
#   k, m, delta, beta
#       always None. Vestigial, and `params` carries the fitted values.
#   _constructed_fit_state
#       rebuilt rather than carried: model_from_dict constructs the model with
#       the changepoints it was given, so __init__ takes the snapshot itself.
#       Persisting a derived snapshot would be a second copy to keep in step.
SKIPPED = [
    'opt', 'opt_status', 'opt_status_message', 'optimizer_used',
    'loss_over_iterations', '_fit_lib_path', 't', 'y', 'y_scaled',
    't_seasonality', '_fit_design_matrix', '_data_columns', 'condition_masks',
    'rng', 'k', 'm', 'delta', 'beta', '_constructed_fit_state',
]


def _read_series(stored):
    """A pandas Series back from its `orient='split'` JSON, dates made naive."""
    series = pd.read_json(StringIO(stored), typ='series', orient='split')
    if series.dtype.kind == 'M':
        series = series.dt.tz_localize(None)
    return series


def _regressor_props(props, transform):
    """A regressor's properties with its nested predictor model transformed.

    No Prophet counterpart: since #33 a regressor may carry a whole
    AnalyticProphet of its own, fitted to predict that regressor's future
    values. Serializing has to recurse, or a round-tripped model can fit a
    future frame's trend and then have nothing to put in its regressor column.
    """
    out = deepcopy({k: v for k, v in props.items() if k != 'predictor'})
    predictor = props.get('predictor')
    out['predictor'] = None if predictor is None else transform(predictor)
    return out


def model_to_dict(model: AnalyticProphet) -> dict:
    """Convert a fitted model to a dictionary suitable for JSON serialization.

    [fc] prophet.serialize.model_to_dict. Model must be fitted. Skips the
    optimizer's own objects, which a prediction never reads.

    Can be reversed with model_from_dict.
    """
    if model.params is None:
        raise ValueError(
            "This can only be used to serialize models that have already been fit."
        )

    model_dict = {
        attribute: getattr(model, attribute) for attribute in SIMPLE_ATTRIBUTES
    }
    # numpy scalars are not JSON-native, and sigma_obs and y_scale are both
    # np.float64 off the optimizer rather than floats
    for attribute, value in model_dict.items():
        if isinstance(value, np.generic):
            model_dict[attribute] = value.item()

    for attribute in PD_SERIES:
        value = getattr(model, attribute)
        model_dict[attribute] = None if value is None else value.to_json(
            orient='split', date_format='iso')
    for attribute in PD_DATAFRAME:
        value = getattr(model, attribute)
        model_dict[attribute] = None if value is None else value.to_json(
            orient='table', index=False)
    for attribute in NP_ARRAY:
        value = getattr(model, attribute)
        model_dict[attribute] = None if value is None else np.asarray(value).tolist()
    for attribute in ORDEREDDICT:
        source = getattr(model, attribute)
        if attribute == 'extra_regressors':
            source = OrderedDict(
                (name, _regressor_props(props, model_to_dict))
                for name, props in source.items())
        model_dict[attribute] = [list(source.keys()), dict(source)]

    # params: {name -> np.ndarray}, [fc] Prophet's own handling
    for attribute in SET_ATTRIBUTES:
        model_dict[attribute] = sorted(getattr(model, attribute))
    model_dict['params'] = {k: np.asarray(v).tolist() for k, v in model.params.items()}
    # the layout is derived from three counts; storing those rebuilds it
    # exactly and keeps the file readable
    model_dict['layout'] = [model.layout.n_changepoints,
                            model.layout.n_seasonality_columns,
                            model.layout.n_holiday_columns]
    model_dict['__analytic_prophet_version'] = __version__
    return model_dict


def model_to_json(model: AnalyticProphet) -> str:
    """Serialize a fitted model to a JSON string.

    [fc] prophet.serialize.model_to_json. Can be deserialized with
    model_from_json.
    """
    return json.dumps(model_to_dict(model))


def model_from_dict(model_dict: dict) -> AnalyticProphet:
    """Recreate a model from a dictionary made by model_to_dict.

    [fc] prophet.serialize.model_from_dict.
    """
    # Constructed *with* the changepoints it was given, rather than plain and
    # patched afterwards. __init__ takes the snapshot `_reset_fit_state` restores
    # from (#41), and that snapshot has to hold the user's list -- a plain
    # instance records None there, so a restored model with explicit
    # changepoints raised `TypeError: object of type 'NoneType' has no len()`
    # on its next fit.
    changepoints = None
    if model_dict['specified_changepoints'] and model_dict['changepoints'] is not None:
        changepoints = _read_series(model_dict['changepoints'])
    model = AnalyticProphet(changepoints=changepoints)

    for attribute in SIMPLE_ATTRIBUTES:
        setattr(model, attribute, model_dict[attribute])
    for attribute in PD_SERIES:
        stored = model_dict[attribute]
        if stored is None:
            setattr(model, attribute, None)
            continue
        setattr(model, attribute, _read_series(stored))
    for attribute in PD_DATAFRAME:
        stored = model_dict[attribute]
        setattr(model, attribute, None if stored is None else pd.read_json(
            StringIO(stored), typ='frame', orient='table', convert_dates=['ds']))
    for attribute in NP_ARRAY:
        stored = model_dict[attribute]
        setattr(model, attribute, None if stored is None else np.array(stored))
    for attribute in ORDEREDDICT:
        key_list, unordered = model_dict[attribute]
        restored = OrderedDict()
        for key in key_list:
            props = unordered[key]
            if attribute == 'extra_regressors':
                props = _regressor_props(props, model_from_dict)
            restored[key] = props
        setattr(model, attribute, restored)

    for attribute in SET_ATTRIBUTES:
        setattr(model, attribute, set(model_dict[attribute]))
    model.params = {k: np.array(v) for k, v in model_dict['params'].items()}
    model.layout = ParameterLayout(*model_dict['layout'])

    # Skipped on the way out, and reset rather than left at whatever the fresh
    # constructor happened to produce.
    model.opt = None
    model.loss_over_iterations = None
    model._fit_lib_path = None
    return model


def model_from_json(model_json: str) -> AnalyticProphet:
    """Deserialize a model from a JSON string made by model_to_json.

    [fc] prophet.serialize.model_from_json.
    """
    return model_from_dict(json.loads(model_json))
