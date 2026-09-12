"""
Shared fixtures for the analytic-prophet test suite.

These currently exercise the pure-Python reference implementation in
CustomProphet (legacy/customProphet.py). Once the C++ core is
consolidated into a single implementation, a second fixture here will
load the compiled extension directly, and the parity tests will compare
against *that* instead of just the Python reference.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "legacy"))
from customProphet import CustomProphet, N_CHANGE_POINTS, n_yearly  # noqa: E402

DATA_PATH = Path(__file__).parent / "data" / "peyton_manning.csv"
PARAM_SIZE = 2 + N_CHANGE_POINTS + 2 * n_yearly  # k, m, delta, beta -> 47


@pytest.fixture
def peyton_manning_df():
    return pd.read_csv(DATA_PATH)


@pytest.fixture
def prepared_model(peyton_manning_df):
    """A CustomProphet with data loaded and preprocessed but not yet fit --
    gives direct access to t_scaled / change_points / normalized_y without
    paying for a full optimize() run in every test."""
    model = CustomProphet()
    model.y = peyton_manning_df["y"].values
    model.ds = pd.to_datetime(peyton_manning_df["ds"])
    model.t_scaled = np.array(
        (model.ds - model.ds.min()) / (model.ds.max() - model.ds.min())
    )
    model.T = peyton_manning_df.shape[0]
    model.scale_period = (model.ds.max() - model.ds.min()).days
    model._normalize_y()
    model._generate_change_points()
    return model


@pytest.fixture
def param_size():
    return PARAM_SIZE


@pytest.fixture
def random_params(param_size):
    """A point in parameter space away from delta=0, so the plain
    numerical-gradient check lands in a smooth region. The kink itself
    (delta=0 exactly) gets its own dedicated test."""
    rng = np.random.default_rng(seed=0)
    return rng.normal(scale=0.5, size=param_size)
