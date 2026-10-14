"""Fit modified copies of Prophet's own Stan program, the way Prophet does (#183).

Four experiments need Prophet's program changed rather than called: the
smooth-surrogate control (#168), the fixed-sigma halves of the multi-start
(#169) and the tau path (#171), and the fix restated inside Stan (#202). Stan cannot hold a parameter fixed at run time
-- it has to move into the `data` block, which is a different program -- so all
three go through here rather than each compiling Stan its own way.

**The programs** live in `paper/stan/`, each a minimal diff from Prophet's
`prophet.stan`, which sits beside them verbatim so the diff is the
documentation:

    original             Prophet's program, unmodified
    fixed_sigma          sigma_obs moved from `parameters` to `data`
    smooth               |delta| replaced by sqrt(delta^2 + epsilon^2) - epsilon,
                         epsilon passed as data
    fixed_sigma_smooth   both
    hadamard             delta = delta_u .* delta_v, the Laplace prior replaced
                         by (|delta_u|^2 + |delta_v|^2) / (2 tau): smooth,
                         unconstrained, the same minimisers in delta (#202)
    split                delta = delta_pos - delta_neg, both <lower=0>, the
                         prior linear in them: the paper's reformulation, as
                         Stan has to state it (#202)
    fixed_sigma_hadamard, fixed_sigma_split
                         each with sigma_obs as data

The smoothed penalty is shifted by epsilon so it is zero at zero and never
exceeds |delta|; it tends to the original's lp__ as epsilon -> 0, which is what
lets #168 score both by either ruler.

**Everything but the program is Prophet's.** The data and the starting point
are captured from a real `Prophet.fit` on the series, and `fit` below repeats
`CmdStanPyBackend.fit`: Newton below 100 observations and L-BFGS above, 10^4
iterations, cmdstanpy's default tolerances, and the fall-back to Newton when
L-BFGS terminates abnormally.

**The CmdStan is the one Prophet bundles**, 2.37.0, asserted at run time rather
than assumed: another version changes the math library and possibly the
optimizer, and then #168 no longer varies only the smoothness.
`tests/test_paper_stan_variants.py` checks that the unmodified program compiled
here reproduces Prophet's own binary before any variant is trusted.
"""
import hashlib
import logging
import os
import shutil
from pathlib import Path

import numpy as np

PAPER = Path(__file__).resolve().parent.parent
STAN = PAPER / "stan"
CMDSTAN_VERSION = "2.37.0"

VARIANTS = {
    "original": "prophet.stan",
    "fixed_sigma": "prophet_fixed_sigma.stan",
    "smooth": "prophet_smooth.stan",
    "fixed_sigma_smooth": "prophet_fixed_sigma_smooth.stan",
    "hadamard": "prophet_hadamard.stan",
    "split": "prophet_split.stan",
    "fixed_sigma_hadamard": "prophet_fixed_sigma_hadamard.stan",
    "fixed_sigma_split": "prophet_fixed_sigma_split.stan",
}
FIXES_SIGMA = {"fixed_sigma", "fixed_sigma_smooth", "fixed_sigma_hadamard",
               "fixed_sigma_split"}
SMOOTHS = {"smooth", "fixed_sigma_smooth"}
HADAMARDS = {"hadamard", "fixed_sigma_hadamard"}
SPLITS = {"split", "fixed_sigma_split"}

# CmdStanPyBackend.fit, restated rather than imported: it is a method that
# also calls the model, and the constants are what has to match.
NEWTON_BELOW = 100
ITERATIONS = int(1e4)


class CmdStanMissing(RuntimeError):
    """No CmdStan installation that can compile a program."""


def cmdstan_path():
    """The full CmdStan installation the variants compile with.

    **Not `cmdstanpy.cmdstan_path()`.** Constructing a `Prophet` points
    cmdstanpy at the CmdStan inside the prophet wheel, which runs Prophet's
    model but holds none of the sources a build needs -- and reports 2.37.0, so
    a version check alone would let it through. After any `Prophet()` in the
    process, cmdstanpy's answer is that copy. So the installation is found here
    instead: `$CMDSTAN`, else cmdstanpy's default location for this version, and
    it has to have the sources.

    Raises `CmdStanMissing` when there is none, and RuntimeError for any version
    but the one Prophet bundles.
    """
    candidates = [Path(os.environ["CMDSTAN"])] if os.environ.get("CMDSTAN") else []
    candidates.append(Path.home() / ".cmdstan" / f"cmdstan-{CMDSTAN_VERSION}")
    for path in candidates:
        if not (path / "src" / "cmdstan" / "main.cpp").exists():
            continue
        version = _makefile_version(path)
        if version != CMDSTAN_VERSION:
            raise RuntimeError(
                f"CmdStan at {path} is {version}, but Prophet bundles {CMDSTAN_VERSION}; "
                f"a different version changes the math library, so a variant would "
                f"differ from Prophet's program by more than its diff")
        return path
    raise CmdStanMissing(
        f"no CmdStan {CMDSTAN_VERSION} installation that can compile (looked in "
        f"{', '.join(str(path) for path in candidates)}); install it with "
        f"cmdstanpy.install_cmdstan(version='{CMDSTAN_VERSION}')")


def _makefile_version(path):
    for line in (Path(path) / "makefile").read_text().splitlines():
        if line.startswith("CMDSTAN_VERSION"):
            return line.split(":=")[1].strip()
    return None


def _build_directory():
    from analytic_prophet.build import cache_root

    return cache_root() / "stan" / CMDSTAN_VERSION


def model(variant):
    """The compiled variant, built once per source and cached.

    Keyed by a hash of the source, so editing a program recompiles it and a
    stale executable is never run. Compiled in the cache rather than beside the
    source, which keeps build products out of the repository.
    """
    import cmdstanpy

    installation = cmdstan_path()
    source = STAN / VARIANTS[variant]
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    directory = _build_directory() / f"{source.stem}-{digest}"
    directory.mkdir(parents=True, exist_ok=True)
    stan_file = directory / source.name
    if not stan_file.exists():
        shutil.copyfile(source, stan_file)
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    # cmdstanpy compiles with whatever its global path says; point it at the
    # full installation for the build and give Prophet its own back afterwards
    try:
        previous = cmdstanpy.cmdstan_path()
    except ValueError:
        previous = None
    cmdstanpy.set_cmdstan_path(str(installation))
    try:
        return cmdstanpy.CmdStanModel(stan_file=str(stan_file))
    finally:
        if previous:
            cmdstanpy.set_cmdstan_path(previous)


def prophet_inputs(prophet_model, df, **fit_kwargs):
    """Fit `prophet_model` on `df`; return (data, inits) exactly as it passed them.

    Captured at `CmdStanModel.optimize`, as `benchmark/_prophet_bridge.py` does
    for the data alone, so a variant starts from the same point on the same
    dictionary. The Prophet model is left fitted, which is what the parity
    check compares against.
    """
    from cmdstanpy import CmdStanModel

    captured = {}
    original = CmdStanModel.optimize

    def spy(self, **kwargs):
        captured.setdefault("data", kwargs.get("data"))
        captured.setdefault("inits", kwargs.get("inits"))
        return original(self, **kwargs)

    CmdStanModel.optimize = spy
    try:
        prophet_model.fit(df, **fit_kwargs)
    finally:
        CmdStanModel.optimize = original
    if "data" not in captured or captured["inits"] is None:
        raise RuntimeError("could not intercept what Prophet passes to Stan; "
                           "cmdstanpy's optimize() signature may have changed")
    return captured["data"], captured["inits"]


def hadamard_factors(delta, tau=None):
    """(delta_u, delta_v) with delta_u * delta_v = delta and |delta_u| = |delta_v|.

    That balanced factorisation is where the Hadamard penalty equals |delta| /
    tau, so the variant's density there is Prophet's; a rate at zero factors as
    (0, 0). Given `tau`, a rate at exactly zero gets (sqrt(tau), 0) instead,
    which is what a *starting point* needs: (0, 0) is a stationary point of the
    factorised objective, and Prophet starts every rate there, so a start at
    (0, 0) could never move. (sqrt(tau), 0) keeps the rate at zero and the
    gradient in delta_v non-zero -- at a cost of 0.5 nats per such rate in the
    penalty, which is why it is for starting points only (#202).
    """
    delta = np.asarray(delta, dtype=float)
    root = np.sqrt(np.abs(delta))
    u, v = np.sign(delta) * root, root
    if tau is not None:
        u = np.where(delta == 0, np.sqrt(tau), u)
        v = np.where(delta == 0, 0.0, v)
    return u, v


def split_parts(delta, floor):
    """(delta_pos, delta_neg) with difference delta, both raised by `floor`.

    Stan declares them `<lower=0>` and optimizes their logarithms, so neither can
    start at zero; raising both by the same amount leaves delta unchanged. How
    high matters: the gradient through the logarithm shrinks with the part, and
    a floor of tau / 50 leaves Stan's L-BFGS far further short than a floor of
    tau, the prior's own scale, which is the default `inputs_for` uses
    (`paper/experiments/inside_stan.py` measures both, #202).
    """
    delta = np.asarray(delta, dtype=float)
    return np.maximum(delta, 0.0) + floor, np.maximum(-delta, 0.0) + floor


def inputs_for(variant, data, inits, sigma_obs=None, epsilon=None):
    """Prophet's (data, inits), adjusted for what `variant` moves into data, and
    with delta translated into the variant's own parameters."""
    data, inits = dict(data), dict(inits)
    if "delta" in inits and variant in HADAMARDS:
        u, v = hadamard_factors(inits.pop("delta"), float(data["tau"]))
        inits["delta_u"], inits["delta_v"] = u.tolist(), v.tolist()
    elif "delta" in inits and variant in SPLITS:
        positive, negative = split_parts(inits.pop("delta"), float(data["tau"]))  # floor tau
        inits["delta_pos"], inits["delta_neg"] = positive.tolist(), negative.tolist()
    if variant in FIXES_SIGMA:
        if sigma_obs is None or not sigma_obs > 0:
            raise ValueError(f"{variant} holds sigma_obs fixed, so it needs a positive sigma_obs")
        data["sigma_obs"] = float(sigma_obs)
        inits.pop("sigma_obs", None)
    elif sigma_obs is not None:
        raise ValueError(f"{variant} estimates sigma_obs; it cannot be given one")
    if variant in SMOOTHS:
        # At epsilon = 0 the surrogate is |delta| again, and its gradient at
        # Prophet's starting point delta = 0 is 0/0.
        if epsilon is None or not epsilon > 0:
            raise ValueError(f"{variant} needs a positive epsilon")
        data["epsilon"] = float(epsilon)
    elif epsilon is not None:
        raise ValueError(f"{variant} has no epsilon")
    return data, inits


def fit(variant, data, inits, sigma_obs=None, epsilon=None, newton_fallback=True,
        **kwargs):
    """[fc] CmdStanPyBackend.fit, on a variant. Returns (params, CmdStanMLE).

    `params` is keyed as Prophet's are -- k, m, delta, sigma_obs, beta, lp__ --
    with sigma_obs reported back as the value it was held at when the variant
    fixes it. `kwargs` reach `optimize` last, as in Prophet, which is how a
    caller overrides the algorithm or the tolerances.
    """
    compiled = model(variant)
    data, inits = inputs_for(variant, data, inits, sigma_obs, epsilon)
    args = dict(data=data, inits=inits,
                algorithm="Newton" if float(data["T"]) < NEWTON_BELOW else "LBFGS",
                iter=ITERATIONS)
    args.update(kwargs)
    try:
        result = compiled.optimize(**args)
    except RuntimeError:
        if not newton_fallback or args["algorithm"] == "Newton":
            raise
        args["algorithm"] = "Newton"
        result = compiled.optimize(**args)

    values = result.optimized_params_dict
    params = {
        "k": float(values["k"]),
        "m": float(values["m"]),
        "delta": np.array([values[f"delta[{j + 1}]"] for j in range(int(data["S"]))]),
        "beta": np.array([values[f"beta[{j + 1}]"] for j in range(int(data["K"]))]),
        "sigma_obs": float(data["sigma_obs"] if variant in FIXES_SIGMA
                           else values["sigma_obs"]),
        "lp__": float(values["lp__"]),
    }
    for name in ("delta_u", "delta_v", "delta_pos", "delta_neg"):
        if f"{name}[1]" in values:
            params[name] = np.array([values[f"{name}[{j + 1}]"] for j in range(int(data["S"]))])
    return params, result


def log_prob(variant, data, params, sigma_obs=None, epsilon=None):
    """A variant's lp__ and gradient at `params`, at full precision.

    Returns (lp__, frame) with the frame as cmdstanpy gives it, columns
    lp__, g_k, g_m, g_delta.1, ... -- the gradient over the variant's own
    parameters, so without g_sigma_obs when sigma is data.
    """
    data, _ = inputs_for(variant, data, {}, sigma_obs, epsilon)
    point = {"k": float(params["k"]), "m": float(params["m"]),
             "delta": [float(v) for v in params["delta"]],
             "beta": [float(v) for v in params["beta"]]}
    if variant in HADAMARDS:
        # the factors given, or the balanced ones, where the density is Prophet's
        u, v = ((params["delta_u"], params["delta_v"]) if "delta_u" in params
                else hadamard_factors(point["delta"]))
        del point["delta"]
        point["delta_u"], point["delta_v"] = [float(x) for x in u], [float(x) for x in v]
    elif variant in SPLITS:
        positive, negative = ((params["delta_pos"], params["delta_neg"])
                              if "delta_pos" in params
                              else split_parts(point["delta"], float(data["tau"])))  # floor tau
        del point["delta"]
        point["delta_pos"] = [float(x) for x in positive]
        point["delta_neg"] = [float(x) for x in negative]
    if variant not in FIXES_SIGMA:
        point["sigma_obs"] = float(params["sigma_obs"])
    frame = model(variant).log_prob(params=point, data=data, jacobian=False,
                                    sig_figs=18)
    return float(frame["lp__"].iloc[0]), frame
