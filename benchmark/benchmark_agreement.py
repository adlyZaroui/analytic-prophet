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
    from customProphet import CustomProphet

    prophet_model = Prophet(**common.PROPHET_KWARGS)
    stan_model, stan_data, prophet_params = bridge.capture_stan_model(prophet_model, df)
    reported = float(np.asarray(prophet_model.params["lp__"]).ravel()[0])
    lp_prophet = bridge.validate_bridge(stan_model, stan_data, prophet_params, reported)

    changepoints_t = np.asarray(stan_data["t_change"], dtype=float)
    X_stan = np.asarray(stan_data["X"], dtype=float)

    # our fit, on Prophet's changepoints so delta indexes the same breakpoints
    ours = CustomProphet()
    ours.set_changepoints = lambda: setattr(ours, "changepoints_t", changepoints_t.copy())
    ours.fit_cpp(df, lib_path=lib_path)

    X_ours = np.ascontiguousarray(ours.make_all_seasonality_features(df)[0].to_numpy(dtype=float))
    beta_ours = ours.params["beta"][0]
    beta_in_stan, residual = bridge.transfer_seasonality(beta_ours, X_ours, X_stan)

    lp_ours = bridge.stan_log_prob(stan_model, stan_data,
                                   ours.params["k"][0][0], ours.params["m"][0][0],
                                   ours.params["delta"][0],
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
        # additive_terms is Prophet's total seasonality across every selected
        # component, which is what our single `seasonality` column holds. Named
        # components were compared individually while only yearly existed.
        "seasonality": relative(prophet_forecast["additive_terms"].values,
                                our_forecast["seasonality"].values),
    }


def short_series_comparison(df, lib_path):
    """Both of Prophet's algorithms against both of ours.

    [fc] the backend runs Newton below 100 observations and L-BFGS at or above.
    Both paths here follow that rule (#25), so the column that matters for a
    user is `rule` -- what a default fit actually produces. `lbfgs` is what the
    rule gives up at these sizes, and printing the two together is what keeps
    that cost measured rather than remembered.
    """
    from prophet import Prophet
    from customProphet import CustomProphet

    scores = {}
    for algorithm in ("Newton", "LBFGS"):
        prophet_model = Prophet(**common.PROPHET_KWARGS)
        stan_model, stan_data, params = bridge.capture_stan_model(
            prophet_model, df, algorithm=algorithm)
        reported = float(np.asarray(prophet_model.params["lp__"]).ravel()[0])
        scores[algorithm] = (stan_model, stan_data,
                             bridge.validate_bridge(stan_model, stan_data, params, reported))

    stan_model, stan_data, lp_newton = scores["Newton"]
    changepoints_t = np.asarray(stan_data["t_change"], dtype=float)

    def ours(**fit_kwargs):
        model = CustomProphet(n_changepoints=len(changepoints_t))
        model.set_changepoints = lambda: setattr(
            model, "changepoints_t", changepoints_t.copy())
        model.fit_cpp(df, lib_path=lib_path, **fit_kwargs)
        return model.optimizer_used, bridge.stan_log_prob(
            stan_model, stan_data, model.params["k"][0][0], model.params["m"][0][0],
            model.params["delta"][0], model.sigma_obs, model.params["beta"][0])

    picked, lp_rule = ours()
    _, lp_lbfgs = ours(algorithm="LBFGS")
    return lp_newton, scores["LBFGS"][2], picked, lp_rule, lp_lbfgs


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
        print()
        print("3. SHORT SERIES -- Prophet's algorithm rule, and what following it costs")
        print("   Newton below 100 observations, L-BFGS at or above (#25).")
        print("   `rule` is the default fit; `lbfgs` is what the rule gives up.")
        print()
        print(f"{'T':>6} {'picks':>7} {'p.newton':>12} {'p.lbfgs':>12} "
              f"{'rule':>12} {'lbfgs':>12} {'rule-newton':>13} {'rule-lbfgs':>12}")
        print("-" * 92)
        for size in (20, 30, 50, 75, 99, 150):
            df = common.load_data(size)
            lp_newton, lp_lbfgs, picked, lp_rule, lp_ours = short_series_comparison(
                df, lib_path)
            print(f"{size:>6} {picked:>7} {lp_newton:12.5f} {lp_lbfgs:12.5f} "
                  f"{lp_rule:12.5f} {lp_ours:12.5f} {lp_rule - lp_newton:+13.5f} "
                  f"{lp_rule - lp_ours:+12.5f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
