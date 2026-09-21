"""Scoring this implementation and Prophet on a common footing.

Raw parameters cannot be compared between the two. Prophet builds its Fourier
features from days since the 1970 epoch, this implementation from days since
the series start, so `beta` lives in a rotated basis -- coefficients differ,
including in sign, while describing the same function. Changepoints also sit
at different locations by default, so `delta` is not comparable element-wise.

What *is* comparable is Stan's log density. It is one scalar, invariant to
those reparameterizations, and it is the quantity the model is defined by.
`CmdStanModel.log_prob` evaluates it at arbitrary parameter values, so both
implementations can be scored by the original's own objective.
"""
import numpy as np


def capture_stan_model(prophet_model, df):
    """Fit `prophet_model` and return (cmdstan_model, stan_data, params).

    The data Prophet hands to Stan is not exposed on the fitted object, so it
    is intercepted at the `CmdStanModel.optimize` call. That couples this to
    cmdstanpy's internals; it is confined to this function deliberately, and
    `validate_bridge` below checks the interception actually worked rather
    than trusting it.
    """
    from cmdstanpy import CmdStanModel

    captured = {}
    original = CmdStanModel.optimize

    def spy(self, **kwargs):
        captured["data"] = kwargs.get("data")
        captured["model"] = self
        return original(self, **kwargs)

    CmdStanModel.optimize = spy
    try:
        prophet_model.fit(df)
    finally:
        CmdStanModel.optimize = original

    if "data" not in captured:
        raise RuntimeError("could not intercept the data Prophet passes to Stan; "
                           "cmdstanpy's optimize() signature may have changed")

    params = {k: np.asarray(v).ravel()
              for k, v in prophet_model.params.items()
              if k in ("k", "m", "delta", "sigma_obs", "beta")}
    return captured["model"], captured["data"], params


def stan_log_prob(stan_model, stan_data, k, m, delta, sigma_obs, beta):
    """Stan's log density at a parameter point. Higher is better.

    jacobian=False matches what `optimize` maximizes -- Stan applies no change
    of variables adjustment for point estimation.
    """
    result = stan_model.log_prob(
        params={"k": float(k), "m": float(m),
                "delta": [float(v) for v in delta],
                "sigma_obs": float(sigma_obs),
                "beta": [float(v) for v in beta]},
        data=stan_data, jacobian=False)
    return float(result["lp__"].iloc[0])


def transfer_seasonality(beta_from, X_from, X_to):
    """Re-express seasonality coefficients in another Fourier basis.

    Both bases span the same space -- same frequencies, same period, differing
    only in time origin and column order -- so transferring the seasonality
    *vector* and solving for coefficients is exact. Returns the coefficients
    and the residual of the solve, which the caller should check is ~0.
    """
    seasonality = X_from @ beta_from
    beta_to, *_ = np.linalg.lstsq(X_to, seasonality, rcond=None)
    residual = float(np.max(np.abs(X_to @ beta_to - seasonality)))
    return beta_to, residual


def validate_bridge(stan_model, stan_data, prophet_params, reported_lp):
    """Check the scoring path reproduces Prophet's own reported lp__.

    If this fails, every comparison built on it is meaningless, so it runs
    before any of them.
    """
    scored = stan_log_prob(stan_model, stan_data,
                           prophet_params["k"][0], prophet_params["m"][0],
                           prophet_params["delta"], prophet_params["sigma_obs"][0],
                           prophet_params["beta"])
    if not np.isclose(scored, reported_lp, rtol=1e-9, atol=1e-6):
        raise RuntimeError(
            f"scoring Prophet's own parameters gives {scored}, but it reported "
            f"{reported_lp}; the bridge is not measuring what it claims to")
    return scored
