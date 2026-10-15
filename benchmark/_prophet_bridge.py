"""Scoring this implementation and Prophet on a common footing.

The parameters are comparable now: both build their Fourier features from days
since the 1970 epoch (#36), and both place the changepoints by Prophet's rule
(#15). They still differ wherever Prophet stops short of the optimum, by an
amount that depends on how flat the objective is in each direction, so a
tolerance on them would measure that flatness rather than either fit.

What settles it is Stan's log density. It is one scalar, the quantity the model
is defined by, and `CmdStanModel.log_prob` evaluates it at arbitrary parameter
values, so both implementations can be scored by the original's own objective.
"""
import numpy as np


def capture_stan_model(prophet_model, df, **fit_kwargs):
    """Fit `prophet_model` and return (cmdstan_model, stan_data, params).

    `fit_kwargs` reach `CmdStanModel.optimize` through Prophet's own
    `args.update(kwargs)`, which is how `algorithm='Newton'` is forced -- see
    #25, where the two algorithms have to be compared on the same data.

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
        prophet_model.fit(df, **fit_kwargs)
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

    Both bases span the same space -- same frequencies, same period -- so
    transferring the seasonality *vector* and solving for coefficients is
    exact. Returns the coefficients and the residual of the solve, which the
    caller should check is ~0. Since #36 the two bases are the same one, so
    this is the identity up to rounding; going through the seasonality rather
    than copying `beta` keeps a basis change on either side from passing
    silently.
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
