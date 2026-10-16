"""
Issue #240: how far the default fit stops from the converged optimum.

#208 found Prophet's own Newton scoring above the corrected optimum on 15 of
200 census series. The corrected optimum is this implementation's default fit,
which Stan's own stopping tests end a little short of exact optimality.
`paper/experiments/converged.py` measures by how much, on the same 200 series:
it carries each default fit on with the compiled projected Newton method.

The decision, recorded in docs/deviations.md: the default fit keeps Stan's
tests. The gap is a few ten-thousandths of a nat, closing it costs about a
tenth of the fit's time, and Prophet's own fit runs the same tests. The paper
measures from the converged point where a comparison is that fine.
Checked here:

  * the results are complete, on the sample the rest of the paper uses;
  * what section 5.2 says of them holds, and the one series it counts out
    is one Stan's relative test stopped;
  * against the converged point Prophet's Newton stays below on every series,
    and the 15 series #240 found are the ones where it beats the default fit;
  * the Newton run from the default fit never makes it worse, live;
  * docs/deviations.md records the gap with its number.
"""
import csv
import importlib.util
import statistics
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).parent.parent
PAPER = REPO / "paper"
EXPERIMENTS = PAPER / "experiments"
RESULTS = PAPER / "results"
STAN_RELATIVE = 1e4 * np.finfo(float).eps        # tol_rel_obj, as Stan scales it
WORDS = "zero one two three four five six seven eight nine ten".split()


def _module(name):
    if str(EXPERIMENTS) not in sys.path:
        sys.path.insert(0, str(EXPERIMENTS))
    spec = importlib.util.spec_from_file_location(name, EXPERIMENTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def rows():
    return _module("converged").read()


@pytest.fixture(scope="module")
def macros():
    return _module("quote_numbers").macros()


def _text(name):
    return " ".join((PAPER / "sections" / f"{name}.tex").read_text().split())


def test_the_run_is_complete_and_on_the_rounding_sample(rows):
    with open(RESULTS / "rounding.csv", newline="") as handle:
        rounding = {row["series"] for row in csv.DictReader(handle)}
    assert {row["series"] for row in rows} == rounding
    assert len(rows) == _module("converged").SAMPLE


def test_what_section_5_2_says_of_it_holds(rows, macros):
    text = _text("evidence")
    # the default fit stops short of the optimum, and the Newton run takes it there
    assert all(row["gain"] >= 0 for row in rows)
    assert statistics.median(row["relaxed_default"] for row in rows) > 1e-2
    assert statistics.median(row["relaxed_converged"] for row in rows) < 1e-6
    # "two steps on N of them, at most nine": counted from the loss trace
    steps = [row["steps_converged"] for row in rows]
    assert macros["ConvergedSteps"] == WORDS[statistics.mode(steps)]
    assert macros["ConvergedStepsSeries"] == f"{steps.count(statistics.mode(steps))}"
    assert macros["ConvergedStepsMax"] == WORDS[max(steps)]
    assert min(steps) >= 1, "a run that took no step measures nothing"
    assert r"\ConvergedSteps{} steps on \ConvergedStepsSeries{} of them" in text
    # "below 10^{-3} on all but one": the uncertified are counted, not hidden
    uncertified = [row for row in rows if row["relaxed_converged"] >= 1e-3]
    assert macros["ConvergedUncertified"] == WORDS[len(uncertified)]
    assert r"below $10^{-3}$ on all but \ConvergedUncertified" in text
    # and docs/deviations.md's account of them: Stan's relative test ended the
    # run, its whole gain under the threshold at that density
    for row in uncertified:
        assert row["gain"] < STAN_RELATIVE * abs(row["lp_converged"]), row["series"]
    # it never loses an exact zero
    assert all(row["exact_zeros_converged"] >= row["exact_zeros_default"] for row in rows)


def test_prophets_newton_against_the_converged_optimum(rows, macros):
    """What #240 found, resolved. From the default fit Prophet's Newton scored
    higher on 15 series; from the converged point it is below on all, and the
    15 are what section 7 counts as closer than this implementation's fit."""
    optimum = {row["series"]: row["lp_converged"] for row in rows}
    default = {row["series"]: row["lp_default"] for row in rows}
    census = _module("newton_census").read()
    assert all(optimum[row["series"]] > row["lp_newton"] for row in census)
    # newton_census.py's reference is the same default fit, to the last bit
    assert all(row["lp_optimum"] == default[row["series"]] for row in census)
    closer = [row["lp_newton"] - default[row["series"]] for row in census
              if row["lp_newton"] > default[row["series"]]]
    assert macros["NewtonCensusAbove"] == f"{len(closer)}"
    assert macros["NewtonCensusAboveMax"] == f"{max(closer):.2g}"
    text = _text("discussion")
    assert r"closer than that on \NewtonCensusAbove{} of the series" in text
    # and the default fit stops far closer than Prophet's Newton at the median
    newton = statistics.median(optimum[row["series"]] - row["lp_newton"] for row in census)
    assert newton > 5 * statistics.median(row["gain"] for row in rows)


def test_the_newton_run_from_the_default_fit_never_makes_it_worse(peyton_manning_df,
                                                                  compiled_optimizer_module):
    """Live, on Peyton Manning at T = 1000: the run converged.py makes."""
    from analytic_prophet import AnalyticProphet

    df = peyton_manning_df.iloc[:1000].reset_index(drop=True)
    default = AnalyticProphet()
    default.fit(df, lib_path=compiled_optimizer_module)
    start = {"k": float(default.params["k"][0][0]), "m": float(default.params["m"][0][0]),
             "delta": np.asarray(default.params["delta"][0], dtype=float),
             "sigma_obs": float(default.sigma_obs),
             "beta": np.asarray(default.params["beta"][0], dtype=float)}
    carried = AnalyticProphet()
    carried.fit(df, lib_path=compiled_optimizer_module, algorithm="Newton", initial_params=start)
    before = default._minus_log_posterior(default.get_parameters())
    after = carried._minus_log_posterior(carried.get_parameters())
    assert after <= before
    steps = len(carried.opt.loss_trace) - 1 - carried.opt.saddle_escapes
    assert 1 <= steps <= 9


def _short(value):
    return f"{value:.1e}".replace("e-0", "e-")


def test_the_decision_is_recorded_with_its_numbers(rows):
    """#240's acceptance: docs/deviations.md records the gap with the number.
    Its numbers are typed, so each is held to the recorded results."""
    deviations = " ".join((REPO / "docs" / "deviations.md").read_text().split())
    section = deviations[deviations.index("## What Stan's tests leave on the compiled path"):]
    gain = [row["gain"] for row in rows]
    assert f"short of the converged point, median | **{_short(statistics.median(gain))} nats**" \
        in section
    assert f"**{_short(max(gain))} nats**" in section
    relaxed = (statistics.median(row["relaxed_default"] for row in rows),
               statistics.median(row["relaxed_converged"] for row in rows))
    assert f"| {relaxed[0]:.2g} | **{_short(relaxed[1])}** |" in section
    steps = [row["steps_converged"] for row in rows]
    usual = statistics.mode(steps)
    assert (f"takes {WORDS[usual]} steps on {steps.count(usual)} of the series and at most "
            f"{WORDS[max(steps)]}") in section
    same = sum(1 for row in rows if row["exact_zeros_converged"] == row["exact_zeros_default"])
    more = sum(1 for row in rows if row["exact_zeros_converged"] > row["exact_zeros_default"])
    assert f"the same on {same} series, more on {more}, fewer on none" in section
    ratio = statistics.median(row["seconds_converged"] / row["seconds_default"] for row in rows)
    assert f"**{round(100 * ratio)}%** at the median" in section
    for row in rows:
        if row["relaxed_converged"] >= 1e-3:
            name = row["series"].removeprefix("m4_daily_").removeprefix("m4_weekly_")
            assert f"{name} (T = {row['observations']})" in section
            assert f"residual of {row['relaxed_converged']:.2g}" in section

    # which test ended the default fit: not the cap, and not tol_grad, which
    # stops at a projected gradient of 1e-8 -- so Stan's relative test
    iterations = max(row["iterations_default"] for row in rows)
    assert iterations < 10_000 and f"at most {iterations:,} of 10,000" in section
    floor = min(row["relaxed_default"] for row in rows)
    assert floor > 1e-4 and f"at least {floor:.2g}" in section

    # the Known differences bullet no longer quotes the sigma_obs-era number as current
    assert "Taking Stan's number would put this path 4.43 nats" not in deviations
    assert "#what-stans-tests-leave-on-the-compiled-path" in deviations
