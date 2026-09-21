"""Agreement with the original Prophet: posterior value and predictions.

This is the acceptance criterion the project is held to (issue #30). It has
two halves, because they answer different questions and can disagree.

  1. POSTERIOR. Is our optimum at least as good as Prophet's, scored by
     Stan's own log density? Both are fitted on the *same* model
     specification -- Prophet's changepoints -- so that this isolates
     optimizer quality rather than mixing in a placement difference.

  2. PREDICTIONS. Do the forecasts agree? Measured on this implementation's
     *default* configuration, because that is what a user gets.

Comparing fitted parameters directly is not one of the criteria, and cannot
be: `beta` lives in a rotated Fourier basis and `delta` is indexed against
different changepoints. See _prophet_bridge.py.

    python benchmark/benchmark_agreement.py
    python benchmark/benchmark_agreement.py --sizes 300 1000 --horizon 90
"""
import argparse
import logging
import sys
import tempfile

import numpy as np

import _common as common
import _prophet_bridge as bridge

logging.getLogger("cmdstanpy").setLevel(logging.ERROR)


def posterior_comparison(df, lib_path):
    """Score both optima under Stan's log density, same model specification."""
    from prophet import Prophet
    from customProphet import (CustomProphet, fourier_components, n_yearly,
                           SIGMA_OBS_IDX, YEARLY_PERIOD)

    prophet_model = Prophet(**common.PROPHET_KWARGS)
    stan_model, stan_data, prophet_params = bridge.capture_stan_model(prophet_model, df)
    reported = float(np.asarray(prophet_model.params["lp__"]).ravel()[0])
    lp_prophet = bridge.validate_bridge(stan_model, stan_data, prophet_params, reported)

    t_change = np.asarray(stan_data["t_change"], dtype=float)
    X_stan = np.asarray(stan_data["X"], dtype=float)

    # our fit, on Prophet's changepoints so delta indexes the same breakpoints
    ours = CustomProphet()
    ours._generate_change_points = lambda: setattr(ours, "change_points", t_change.copy())
    ours.fit_cpp(df, lib_path=lib_path)

    X_ours = fourier_components(ours.t_seasonality, YEARLY_PERIOD, n_yearly)
    beta_ours = ours.opt_params[SIGMA_OBS_IDX + 1:]
    beta_in_stan, residual = bridge.transfer_seasonality(beta_ours, X_ours, X_stan)

    lp_ours = bridge.stan_log_prob(stan_model, stan_data,
                                   ours.opt_params[0], ours.opt_params[1],
                                   ours.opt_params[2:2 + len(t_change)],
                                   ours.sigma_obs, beta_in_stan)
    return lp_prophet, lp_ours, residual


def prediction_comparison(df, lib_path, horizon):
    """Compare forecasts, each implementation in its default configuration."""
    from prophet import Prophet
    from customProphet import CustomProphet

    prophet_model = Prophet(**common.PROPHET_KWARGS)
    prophet_model.fit(df)
    prophet_forecast = prophet_model.predict(prophet_model.make_future_dataframe(periods=horizon))

    ours = CustomProphet()
    ours.fit_cpp(df, lib_path=lib_path)
    our_forecast = ours.predict(ours.make_future_dataframe(periods=horizon))

    n = min(len(prophet_forecast), len(our_forecast))
    y_scale = float(np.max(np.abs(df["y"].values)))

    def relative(a, b):
        d = np.abs(np.asarray(a)[:n] - np.asarray(b)[:n]) / y_scale
        return float(np.max(d)), float(np.mean(d))

    return {
        "yhat": relative(prophet_forecast["yhat"].values, our_forecast["yhat"].values),
        "trend": relative(prophet_forecast["trend"].values, our_forecast["trend"].values),
        "seasonality": relative(prophet_forecast["yearly"].values, our_forecast["seasonality"].values),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", type=int, nargs="+", default=[300, 1000, 2905])
    parser.add_argument("--horizon", type=int, default=30)
    args = parser.parse_args()

    if not common.prophet_available():
        print(common.PROPHET_INSTALL_HINT, file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory() as tmp:
        lib_path = common.build_cpp_extension(tmp)
        if lib_path is None:
            return 1

        print("1. POSTERIOR -- Stan's own log density, higher is better")
        print("   both fitted on Prophet's changepoints, isolating optimizer quality")
        print()
        print(f"{'T':>6} {'prophet':>14} {'ours':>14} {'difference':>13}  verdict")
        print("-" * 72)
        for size in args.sizes:
            df = common.load_data(size)
            lp_prophet, lp_ours, residual = posterior_comparison(df, lib_path)
            verdict = "ours >= prophet" if lp_ours >= lp_prophet else "*** WORSE ***"
            print(f"{size:>6} {lp_prophet:14.4f} {lp_ours:14.4f} {lp_ours - lp_prophet:+13.4f}  {verdict}")
            if residual > 1e-8:
                print(f"       basis transfer residual {residual:.2e} -- comparison unreliable")

        print()
        print(f"2. PREDICTIONS -- default configuration, history + {args.horizon} days")
        print("   as a percentage of the series scale, max|difference| (mean)")
        print()
        print(f"{'T':>6} {'yhat':>18} {'trend':>18} {'seasonality':>18}")
        print("-" * 64)
        for size in args.sizes:
            df = common.load_data(size)
            r = prediction_comparison(df, lib_path, args.horizon)
            cells = "".join(f"{mx * 100:11.3f}% ({mn * 100:.3f}%)" for mx, mn in
                            (r["yhat"], r["trend"], r["seasonality"]))
            print(f"{size:>6} {cells}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
