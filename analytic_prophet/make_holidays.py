"""Holiday frames and their design columns.

[fc] prophet/make_holidays.py, which carries the country lookup, plus the
frame validation and feature construction Prophet keeps on the class.

The `holidays` package is imported lazily and only here: trend, seasonality
and a hand-written holidays frame all work without it.
"""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("analytic_prophet")


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

    [fc] Prophet.make_holiday_features. Returns (features, prior_scales,
    names): a DataFrame with Prophet's own column names, one prior scale per
    column, and the holiday names in the order they were first seen.

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
        return pd.DataFrame(index=range(len(dates))), [], []

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
        return pd.DataFrame(index=range(len(dates))), [], list(prior_scales)

    names = sorted(columns)
    features = pd.DataFrame({name: columns[name] for name in names}, columns=names)
    scales = [prior_scales[name.split("_delim_")[0]] for name in names]
    return features, scales, list(prior_scales)
