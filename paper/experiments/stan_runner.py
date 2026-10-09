"""Fit modified copies of Prophet's own Stan program, the way Prophet does (#183).

Three experiments need Prophet's program changed rather than called: the
smooth-surrogate control (#168) and the fixed-sigma halves of the multi-start
(#169) and the tau path (#171). Stan cannot hold a parameter fixed at run time
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
}
FIXES_SIGMA = {"fixed_sigma", "fixed_sigma_smooth"}
SMOOTHS = {"smooth", "fixed_sigma_smooth"}

# CmdStanPyBackend.fit, restated rather than imported: it is a method that
# also calls the model, and the constants are what has to match.
NEWTON_BELOW = 100
ITERATIONS = int(1e4)


def cmdstan_path():
    """CmdStan's location, refusing any version but the one Prophet bundles."""
    import cmdstanpy

    try:
        path = Path(cmdstanpy.cmdstan_path())
    except ValueError as error:
        raise RuntimeError(
            f"CmdStan {CMDSTAN_VERSION} is not installed; install it with "
            f"cmdstanpy.install_cmdstan(version='{CMDSTAN_VERSION}')") from error
    version = _makefile_version(path)
    if version != CMDSTAN_VERSION:
        raise RuntimeError(
            f"CmdStan at {path} is {version}, but Prophet bundles {CMDSTAN_VERSION}; "
            f"a different version changes the math library, so a variant would "
            f"differ from Prophet's program by more than its diff")
    return path


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
    from cmdstanpy import CmdStanModel

    cmdstan_path()
    source = STAN / VARIANTS[variant]
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    directory = _build_directory() / f"{source.stem}-{digest}"
    directory.mkdir(parents=True, exist_ok=True)
    stan_file = directory / source.name
    if not stan_file.exists():
        shutil.copyfile(source, stan_file)
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    return CmdStanModel(stan_file=str(stan_file))


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


def inputs_for(variant, data, inits, sigma_obs=None, epsilon=None):
    """Prophet's (data, inits), adjusted for what `variant` moves into data."""
    data, inits = dict(data), dict(inits)
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
    if variant not in FIXES_SIGMA:
        point["sigma_obs"] = float(params["sigma_obs"])
    frame = model(variant).log_prob(params=point, data=data, jacobian=False,
                                    sig_figs=18)
    return float(frame["lp__"].iloc[0]), frame
