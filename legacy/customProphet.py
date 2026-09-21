import glob
import importlib
import importlib.util
import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import halfcauchy
from typing import Tuple

# ---------------------------------------------------------------------------
# Model constants, checked against facebook/prophet (additive mode, linear
# growth, MAP). Sources:
#   [stan] python/stan/prophet.stan
#   [fc]   python/prophet/forecaster.py
#
# Three categories, kept deliberately distinct:
#
#   FITTED  free parameters in Stan's `parameters` block, estimated by L-BFGS.
#           Here: k, m, delta(S), sigma_obs, beta(K) -- see the layout below.
#   SCALE   the *scale of a prior* on a fitted parameter. A scale is not the
#           value of anything; it controls how hard a fitted parameter is
#           pulled toward zero. Prophet estimates none of these -- there are no
#           hierarchical priors in the model.
#   FIXED   structural constants (counts, ranges, orders), passed to Stan as
#           data.
# ---------------------------------------------------------------------------

# FIXED -- number of changepoints S. [fc] Prophet.__init__ n_changepoints=25
N_CHANGE_POINTS = 25
# FIXED -- changepoints span the first 80% of history.
# [fc] Prophet.__init__ changepoint_range=0.8
CHANGEPOINT_RANGE = 0.8
# FIXED -- Fourier order N for yearly seasonality, so K = 2N = 20 columns.
# [fc] set_auto_seasonalities, yearly fourier_order=10
n_yearly = 10

# SCALE on delta, the changepoint rate adjustments: delta ~ double_exponential(0, tau).
# [stan] model block; [fc] Prophet.__init__ changepoint_prior_scale=0.05 (user-configurable)
TAU = 0.05
# SCALE on beta, the seasonality coefficients: beta ~ normal(0, sigmas).
# [stan] model block; [fc] Prophet.__init__ seasonality_prior_scale=10.0 (user-configurable)
# NOTE: `sigmas` is vector[K] in Stan, one scale per regressor column, so
# seasonality and holiday terms can differ. A scalar is only adequate because
# this implementation is yearly-seasonality-only; adding holidays or a second
# seasonality means making this per-column.
SIGMA = 10
# SCALE on sigma_obs, the observation noise: sigma_obs ~ normal(0, 0.5),
# truncated at 0 by `real<lower=0>`. [stan] parameters + model blocks.
# Hardcoded in prophet.stan -- not user-configurable.
SIGMA_OBS_PRIOR_SCALE = 0.5
# SCALE on k: k ~ normal(0, 5). [stan] model block. Hardcoded, not configurable.
sigma_k = 5
# SCALE on m: m ~ normal(0, 5). [stan] model block. Hardcoded, not configurable.
sigma_m = 5

# Stan's L-BFGS convergence criteria, with the CmdStan defaults Prophet runs
# under: CmdStanPyBackend.fit calls optimize(algorithm='LBFGS', iter=int(1e4))
# and sets no tolerances or history_size.
# [stan] src/stan/optimization/bfgs.hpp, ConvergenceOptions + step()
#
# Stan stops as soon as ANY of five tests holds: absolute and relative
# objective change, absolute and relative gradient norm, and parameter change.
# L-BFGS-B exposes the first and third of those; the C++ core evaluates four
# of the five directly (see optimize.cpp). Matching them is what stops both
# paths running tens of thousands of iterations past convergence.
STAN_EPS = 2.220446049250313e-16   # machine epsilon, as Stan uses it
STAN_TOL_REL_OBJ = 1e+4            # tol_rel_obj, scaled by STAN_EPS
STAN_TOL_GRAD = 1e-8               # tol_grad
STAN_MAX_ITERATIONS = 10000        # Prophet passes iter=int(1e4)

# INIT -- Prophet overrides Stan's random init with explicit values, so Stan's
# `init_r * N(0, 1)` default is never reached. [fc] calculate_initial_params
# returns sigma_obs=1.0, delta=zeros(S), beta=zeros(K), and k/m from
# linear_growth_init (see below).
SIGMA_OBS_INIT = 1.0

# Parameter vector layout shared by the analytic posterior/gradient and by
# predict()/trend_forecast_uncertainty(): [k, m, delta (S), sigma_obs, beta (2*n_yearly)]
K_IDX = 0
M_IDX = 1
DELTA_SLICE = slice(2, 2 + N_CHANGE_POINTS)
SIGMA_OBS_IDX = DELTA_SLICE.stop
BETA_SLICE = slice(SIGMA_OBS_IDX + 1, SIGMA_OBS_IDX + 1 + 2 * n_yearly)

def linear_growth_init(t_scaled, normalized_y):
    """Prophet's deterministic starting point for (k, m): the line through the
    first and last points of the scaled series.

    [fc] Prophet.linear_growth_init -- it indexes by argmin/argmax of ds rather
    than assuming the frame is sorted, so this does the same via t_scaled.

        k = (y_scaled[i1] - y_scaled[i0]) / (t[i1] - t[i0])
        m = y_scaled[i0] - k * t[i0]

    Prophet always passes this in explicitly, which is why Stan's random
    `init_r * N(0, 1)` default never applies.
    """
    i0 = int(np.argmin(t_scaled))
    i1 = int(np.argmax(t_scaled))
    span = t_scaled[i1] - t_scaled[i0]
    k = (normalized_y[i1] - normalized_y[i0]) / span
    m = normalized_y[i0] - k * t_scaled[i0]
    return float(k), float(m)

def det_dot(a, b):
    return (a * b[None, :]).sum(axis=-1)

# Prophet measures seasonal time in days since the Unix epoch, not since the
# start of the series. [fc] Prophet.fourier_series
FOURIER_EPOCH = pd.Timestamp("1970-01-01")
YEARLY_PERIOD = 365.25

def seasonal_time(ds):
    """Days since the 1970 epoch, the clock Prophet builds Fourier features on.

    Using the series start instead -- as this did before -- shifts the phase of
    every frequency. The fitted seasonality is unchanged (an orthogonal
    rotation per frequency, and `beta`'s prior is rotation-invariant), but the
    coefficients are not comparable with Prophet's, which is what this fixes.
    """
    return (pd.to_datetime(ds) - FOURIER_EPOCH).dt.total_seconds().to_numpy() / (24 * 60 * 60)

def fourier_components(t_days, period, n):
    """Fourier features, column-for-column identical to Prophet's.

    Ordering matters as much as the clock: Prophet interleaves, putting
    sin(order i) at column 2i and cos(order i) at 2i+1. This emitted all
    cosines then all sines, which permutes `beta` even when the phase agrees.
    """
    t_days = np.asarray(t_days, dtype=float)
    angles = (2 * np.pi / period) * np.outer(t_days, np.arange(1, n + 1))
    result = np.empty((t_days.shape[0], 2 * n))
    result[:, 0::2] = np.sin(angles)
    result[:, 1::2] = np.cos(angles)
    return result

def extract_params(params):
    k = params[K_IDX]
    m = params[M_IDX]
    delta = params[DELTA_SLICE]
    sigma_obs = params[SIGMA_OBS_IDX]
    beta = params[BETA_SLICE]
    return k, m, delta, sigma_obs, beta

def from_dict_to_array(params):
    k = np.array([params['k']])
    m = np.array([params['m']])
    delta = params['delta']
    sigma_obs = np.array([params['sigma_obs']])
    beta = np.zeros((2 * 10,))
    return np.concatenate((k, m, delta, sigma_obs, beta))

CPP_MODULE_NAME = 'analytic_prophet_cpp'

BUILD_HINT = (
    "Build it from legacy/optimize.cpp -- see the compile command in that file's "
    "trailing comment, or let tests/conftest.py's compiled_optimizer_module "
    "fixture build it for you."
)

_cpp_module_cache = {}

def load_cpp_module(lib_path=None):
    """Import the compiled pybind11 extension backing fit_cpp().

    With lib_path=None this is an ordinary import, so a built or installed
    extension is found on sys.path like any other module; failing that, it
    looks for one built in place next to this file. Passing lib_path loads a
    specific .so, which is how the tests point at one built into a temp dir.

    The ctypes binding this replaced hardcoded a *relative* path
    ('./liboptimization.so'), so it only worked when the process happened to be
    running from the right directory.
    """
    if lib_path is None:
        try:
            return importlib.import_module(CPP_MODULE_NAME)
        except ImportError:
            here = os.path.dirname(os.path.abspath(__file__))
            candidates = sorted(glob.glob(os.path.join(here, CPP_MODULE_NAME + '*.so')))
            if not candidates:
                raise ImportError(
                    f"{CPP_MODULE_NAME} is not importable and no build of it was found "
                    f"in {here}. {BUILD_HINT}"
                )
            lib_path = candidates[0]

    lib_path = os.path.abspath(lib_path)
    if lib_path not in _cpp_module_cache:
        spec = importlib.util.spec_from_file_location(CPP_MODULE_NAME, lib_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"{lib_path} is not loadable as a Python extension module. {BUILD_HINT}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _cpp_module_cache[lib_path] = module
    return _cpp_module_cache[lib_path]

def canonical_to_cpp(params):
    """(k, m, delta, sigma_obs, beta) -> (k, m, delta, beta, zeta).

    The C++ core carries zeta = log(sigma_obs) as the LAST element, whereas the
    canonical layout keeps sigma_obs between delta and beta, mirroring the order
    of Stan's `parameters` block. Both describe the same model; only the
    packing differs. The log is what keeps sigma_obs positive in the C++, since
    liblbfgs has no box constraints.
    """
    k, m, delta, sigma_obs, beta = extract_params(params)
    return np.concatenate(([k], [m], delta, beta, [np.log(sigma_obs)]))

def cpp_to_canonical(params):
    """Inverse of canonical_to_cpp: sigma_obs = exp(zeta), moved into place."""
    params = np.asarray(params, dtype=float)
    n_delta = N_CHANGE_POINTS
    k, m = params[0], params[1]
    delta = params[2:2 + n_delta]
    beta = params[2 + n_delta:-1]
    sigma_obs = np.exp(params[-1])
    return np.concatenate(([k], [m], delta, [sigma_obs], beta))

def canonical_to_split(params):
    """(k, m, delta, sigma_obs, beta) -> (k, m, delta_pos, delta_neg, sigma_obs, beta).

    The Laplace prior on delta puts |delta|/tau in the objective, which is not
    differentiable at delta=0 -- and that is exactly where the optimum sits,
    since the prior is what drives most changepoint rates to zero. L-BFGS-B
    assumes a smooth objective and stalls on those kinks well short of the
    optimum (while still reporting success).

    Splitting delta into non-negative parts, delta = delta_pos - delta_neg,
    turns |delta| into delta_pos + delta_neg: smooth, with the non-smoothness
    pushed into simple bound constraints L-BFGS-B handles natively. At an
    optimum at most one of each pair is non-zero, so the two problems have the
    same solution. This is the same fix the C++ core gets from OWL-QN.
    """
    k, m, delta, sigma_obs, beta = extract_params(params)
    return np.concatenate(([k], [m], np.maximum(delta, 0), np.maximum(-delta, 0), [sigma_obs], beta))

def split_to_canonical(z, n_delta=N_CHANGE_POINTS):
    """Inverse of canonical_to_split: delta = delta_pos - delta_neg."""
    k, m = z[0], z[1]
    delta_pos = z[2:2 + n_delta]
    delta_neg = z[2 + n_delta:2 + 2 * n_delta]
    sigma_obs = z[2 + 2 * n_delta]
    beta = z[3 + 2 * n_delta:]
    return np.concatenate(([k], [m], delta_pos - delta_neg, [sigma_obs], beta))

def compute_trend(k, m, delta, change_points, t_scaled, y_absmax):
    """Piecewise-linear trend in normalized-y space, de-normalized once at
    the end. Shared by predict() and trend_forecast_uncertainty() so the
    de-normalization can't drift apart between the two again."""
    A = (t_scaled[:, None] >= change_points) * 1
    gamma = -change_points * delta
    trend_normalized = (k + det_dot(A, delta)) * t_scaled + (m + det_dot(A, gamma))
    return trend_normalized * y_absmax

class CustomProphet:
    
    def __init__(self):
        self.rng = np.random.default_rng()
        
        self.t_scaled = None
        self.y = None
        self.normalized_y = None
        self.y_absmax = None
        self.ds = None
        
        self.T = None
        self.n_changepoints = N_CHANGE_POINTS
        self.change_points = None
        self.changepoint_range = CHANGEPOINT_RANGE
        
        self.tau = TAU # sparse prior on rate adjustments delta
        self.sigma = SIGMA # prior on fourier coefficients beta
        self.sigma_obs = SIGMA_OBS_INIT # estimated by fit(); fit_cpp() keeps this fixed

        self.m = None
        self.k = None
        self.delta = None
        self.beta = None

        self.opt_params = None
        self.loss_over_iterations = None
        self.opt = None
        
        self.sigma_k = sigma_k
        self.sigma_m = sigma_m

        self.t_seasonality = None

    def get_parameters(self) -> np.array:
        return self.opt_params
        
    def _normalize_y(self) -> None:
        self.y_absmax = np.max(np.abs(self.y))
        self.normalized_y = np.array(self.y / self.y_absmax)
    
    def _generate_change_points(self) -> None:
        """Changepoints spaced uniformly in *scaled time* over the first
        changepoint_range of history.

        KNOWN DIVERGENCE from [fc] set_changepoints, which spaces them over
        uniformly-spaced row *indices* of the first 80% of history:

            np.linspace(0, hist_size - 1, n_changepoints + 1).round().astype(int)

        and then takes the ds values at those rows. For regularly-spaced daily
        data the two agree; for irregular or gappy series they diverge, since
        index-spacing follows observation density and time-spacing does not.
        Left as-is deliberately -- see the follow-up issue.
        """
        max_t_scaled = np.max(self.t_scaled)
        self.change_points = np.linspace(0, self.changepoint_range * max_t_scaled, self.n_changepoints + 1)[1:]

        
    def _design_matrices(self):
        """The changepoint indicator A (T x S) and the Fourier design matrix
        (T x K).

        Both depend only on t_scaled and change_points, so they are constant
        for a whole fit -- yet rebuilding them was 57% of every objective
        evaluation (issue #28). fit() computes them once and threads them
        through; callers that pass nothing still get correct results, which is
        what keeps the objective usable on its own.
        """
        A = (self.t_scaled[:, None] >= self.change_points) * 1
        x = fourier_components(self.t_seasonality, YEARLY_PERIOD, n_yearly)
        return A, x

    def _minus_log_posterior(self, params: np.array, include_l1_prior: bool=True, design=None) -> float:
        k, m, delta, sigma_obs, beta = extract_params(params)

        A, x = design if design is not None else self._design_matrices()

        # trend component
        gamma = -self.change_points * delta
        g = (k + np.dot(A, delta)) * self.t_scaled + (m + np.dot(A, gamma))

        # seasonality component
        s = np.dot(x, beta)

        y_pred = g + s
        y_true = self.normalized_y

        # T*log(sigma_obs) is the Gaussian normalizing constant -- it can't be
        # dropped now that sigma_obs is itself a parameter being optimized.
        minus_log_posterior = self.T * np.log(sigma_obs) + \
                      np.sum((y_true - y_pred)**2) / (2*sigma_obs**2) + \
                      sigma_obs**2 / (2*SIGMA_OBS_PRIOR_SCALE**2) + \
                      k**2 / (2*self.sigma_k**2) + \
                      m**2 / (2*self.sigma_m**2) + \
                      np.sum(beta**2) / (2*self.sigma**2)

        if include_l1_prior:
            minus_log_posterior += np.sum(np.abs(delta)) / self.tau

        return minus_log_posterior

    def _gradient(self, params: np.array, include_l1_prior: bool=True, design=None) -> np.array:
        k, m, delta, sigma_obs, beta = extract_params(params)

        A, x = design if design is not None else self._design_matrices()

        # trend component
        gamma = -self.change_points * delta
        g = (k + np.dot(A, delta)) * self.t_scaled + (m + np.dot(A, gamma))

        # seasonality component
        s = np.dot(x, beta)

        r = self.normalized_y - g - s

        dk = np.array([-np.sum(r * self.t_scaled) / sigma_obs**2 + k / self.sigma_k**2])
        dm = np.array([-np.sum(r) / sigma_obs**2 + m / self.sigma_m**2])
        ddelta = -np.sum(r[:, None] * (self.t_scaled[:, None] - self.change_points) * A, axis=0) / sigma_obs**2
        dsigma_obs = np.array([self.T / sigma_obs - np.sum(r**2) / sigma_obs**3 + sigma_obs / SIGMA_OBS_PRIOR_SCALE**2])
        dbeta = -np.dot(r, x) / sigma_obs**2 + beta / self.sigma**2

        if include_l1_prior:
            ddelta = ddelta + np.sign(delta) / self.tau

        gradient = np.concatenate([dk, dm, ddelta, dsigma_obs, dbeta])

        return gradient

    def _minus_log_posteriorAndGradient(self, params: np.array, include_l1_prior: bool=True, design=None) -> Tuple[float, np.array]:
        k, m, delta, sigma_obs, beta = extract_params(params)

        A, x = design if design is not None else self._design_matrices()

        # trend component
        gamma = -self.change_points * delta
        g = (k + np.dot(A, delta)) * self.t_scaled + (m + np.dot(A, gamma))

        # seasonality component
        s = np.dot(x, beta)

        r = self.normalized_y - g - s

        minus_log_posterior = self.T * np.log(sigma_obs) + \
                      np.sum(r**2) / (2*sigma_obs**2) + \
                      sigma_obs**2 / (2*SIGMA_OBS_PRIOR_SCALE**2) + \
                      k**2 / (2*self.sigma_k**2) + \
                      m**2 / (2*self.sigma_m**2) + \
                      np.sum(beta**2) / (2*self.sigma**2)

        dk = np.array([-np.sum(r * self.t_scaled) / sigma_obs**2 + k / self.sigma_k**2])
        dm = np.array([-np.sum(r) / sigma_obs**2 + m / self.sigma_m**2])
        ddelta = -np.sum(r[:, None] * (self.t_scaled[:, None] - self.change_points) * A, axis=0) / sigma_obs**2
        dsigma_obs = np.array([self.T / sigma_obs - np.sum(r**2) / sigma_obs**3 + sigma_obs / SIGMA_OBS_PRIOR_SCALE**2])
        dbeta = -np.dot(r, x) / sigma_obs**2 + beta / self.sigma**2

        if include_l1_prior:
            minus_log_posterior += np.sum(np.abs(delta)) / self.tau
            ddelta = ddelta + np.sign(delta) / self.tau

        gradient = np.concatenate([dk, dm, ddelta, dsigma_obs, dbeta])

        return minus_log_posterior, gradient

    # --- split-space (delta = delta_pos - delta_neg) wrappers --------------
    # These are what fit() optimizes over. They evaluate the posterior without
    # its L1 term and add (delta_pos + delta_neg)/tau instead, which is the
    # same value but differentiable -- see canonical_to_split for why.

    def _split_l1_penalty(self, z):
        n_delta = len(self.change_points)
        return np.sum(z[2:2 + 2 * n_delta]) / self.tau

    def _split_minus_log_posterior(self, z: np.array, design=None) -> float:
        smooth = self._minus_log_posterior(split_to_canonical(z), include_l1_prior=False, design=design)
        return smooth + self._split_l1_penalty(z)

    def _split_gradient(self, z: np.array, design=None) -> np.array:
        return self._canonical_gradient_to_split(
            self._gradient(split_to_canonical(z), include_l1_prior=False, design=design))

    def _split_minus_log_posteriorAndGradient(self, z: np.array, design=None) -> Tuple[float, np.array]:
        smooth, gradient = self._minus_log_posteriorAndGradient(
            split_to_canonical(z), include_l1_prior=False, design=design)
        return smooth + self._split_l1_penalty(z), self._canonical_gradient_to_split(gradient)

    def _canonical_gradient_to_split(self, gradient):
        """d/d(delta_pos) = d/d(delta) + 1/tau, d/d(delta_neg) = -d/d(delta) + 1/tau."""
        ddelta = gradient[DELTA_SLICE]
        return np.concatenate((
            gradient[:2],
            ddelta + 1 / self.tau,
            -ddelta + 1 / self.tau,
            [gradient[SIGMA_OBS_IDX]],
            gradient[BETA_SLICE],
        ))

    def fit(self, df: pd.DataFrame, analytic: bool=False, use_combined: bool=False, optimizer: str='L-BFGS-B',
            initial_params: dict=None, fixed_sigma_obs: float=None) -> Tuple[float, float, np.array, np.array]:
        if analytic and use_combined:
            raise ValueError("Both 'analytic' and 'use_combined' cannot be True at the same time.")

        self.y = df['y'].values

        if df['ds'].dtype != 'datetime64[ns]':
            self.ds = pd.to_datetime(df['ds'])
        else:
            self.ds = df['ds']

        self.t_scaled = np.array((self.ds - self.ds.min()) / (self.ds.max() - self.ds.min()))
        self.T = df.shape[0]

        # Calculate the scale period coefficient
        self.t_seasonality = seasonal_time(self.ds)

        self._normalize_y()
        self._generate_change_points()

        # [fc] calculate_initial_params: k/m from linear_growth_init, delta and
        # beta at zero, sigma_obs at 1.0. Prophet passes these to Stan
        # explicitly, so Stan's random init is never used.
        k_init, m_init = linear_growth_init(self.t_scaled, self.normalized_y)
        initial_params_dict = {
            'k': k_init,
            'm': m_init,
            'delta': np.zeros((N_CHANGE_POINTS,)),
            'sigma_obs': SIGMA_OBS_INIT,
            'beta': np.zeros((2 * n_yearly,))
        }
        if initial_params is not None:
            initial_params_dict.update(initial_params)
        if fixed_sigma_obs is not None:
            initial_params_dict['sigma_obs'] = fixed_sigma_obs

        loss_over_iterations = []

        # Built once here, not rebuilt on every evaluation (#28).
        design = self._design_matrices()

        def callback(z):
            fobj = self._minus_log_posterior(split_to_canonical(z), design=design)
            loss_over_iterations.append(fobj)

        initial_params_array = from_dict_to_array(initial_params_dict)

        # sigma_obs must stay positive, mirroring Stan's `real<lower=0> sigma_obs`.
        # fixed_sigma_obs collapses that bound to a single point, pinning sigma_obs
        # for parity with fit_cpp()'s compiled optimizer, which never estimates it.
        sigma_obs_bounds = (fixed_sigma_obs, fixed_sigma_obs) if fixed_sigma_obs is not None else (1e-6, None)
        n_delta = len(self.change_points)
        # Split-space bounds: k, m free; delta_pos/delta_neg >= 0; then sigma_obs, beta
        bounds = [(None, None)] * 2 + [(0, None)] * (2 * n_delta) + [sigma_obs_bounds] + \
                 [(None, None)] * (2 * n_yearly)

        z0 = canonical_to_split(initial_params_array)

        if use_combined:
            objective = lambda z: self._split_minus_log_posteriorAndGradient(z, design=design)
            jac = True
        elif analytic:
            objective = lambda z: self._split_minus_log_posterior(z, design=design)
            jac = lambda z: self._split_gradient(z, design=design)
        else:
            objective = lambda z: self._split_minus_log_posterior(z, design=design)
            jac = None

        options = {'maxiter': STAN_MAX_ITERATIONS}
        if optimizer == 'L-BFGS-B':
            # Stan's iteration cap applies directly. Its tolerances do not
            # transfer as cleanly here as they do in the C++ core, for two
            # measured reasons -- both consequences of this path optimizing the
            # split reformulation rather than Stan's parameterization:
            #
            # gtol is disabled rather than set to Stan's tol_grad. scipy tests
            # the inf-norm of the *projected* gradient, Stan the 2-norm of the
            # full gradient. On the split problem most delta_pos/delta_neg sit
            # at their zero bound, so the projected norm is far smaller than
            # the real one and tol_grad=1e-8 fires at iteration 147 on the
            # 2905-point series, 1.1e-3 short in loss.
            #
            # ftol stays tighter than Stan's tol_rel_obj * eps = 2.22e-12. The
            # split space has long shallow ridges where per-iteration progress
            # falls below that while the fit is still 1.1e-3 from the optimum
            # (measured at T=1000 and T=2905); Stan's own parameterization does
            # not stall there, and the C++ core, which uses it, converges
            # normally under the real tolerance. Loosening this to match Stan
            # numerically would mean a worse fit than Prophet produces, not a
            # closer one.
            options.update({
                'ftol': 1e-16,
                'gtol': 0.0,
                'maxfun': STAN_MAX_ITERATIONS * 10,
            })

        opt_params = minimize(objective,
                        z0,
                        method=optimizer,
                        bounds=bounds,
                        options=options,
                        callback=callback,
                        jac=jac)

        self.opt = opt_params
        self.opt_params = split_to_canonical(opt_params.x)
        self.sigma_obs = self.opt_params[SIGMA_OBS_IDX]
        self.loss_over_iterations = loss_over_iterations
    
    def fit_cpp(self, df: pd.DataFrame, initial_params: dict=None, lib_path: str=None, verbose: bool=False) -> Tuple[float, float, np.array, np.array]:
        """Fit via the compiled C++ core (see load_cpp_module for how it's found)."""
        self.y = df['y'].values

        if df['ds'].dtype != 'datetime64[ns]':
            self.ds = pd.to_datetime(df['ds'])
        else:
            self.ds = df['ds']

        self.t_scaled = np.array((self.ds - self.ds.min()) / (self.ds.max() - self.ds.min()))
        self.T = df.shape[0]

        self.t_seasonality = seasonal_time(self.ds)
        self._normalize_y()
        self._generate_change_points()

        # Same deterministic initialization as fit(), so both fit paths start
        # from the same point. [fc] calculate_initial_params.
        #
        # This used to draw init_r * N(0, 1) with init_r=2.0, described as
        # "STAN initialization" -- but Prophet always passes explicit inits, so
        # Stan's random default is never reached. The draw also came from an
        # unseeded RNG, making fit_cpp() non-reproducible run to run.
        k_init, m_init = linear_growth_init(self.t_scaled, self.normalized_y)
        defaults = {
            'k': k_init,
            'm': m_init,
            'delta': np.zeros((N_CHANGE_POINTS,)),
            'beta': np.zeros((2 * n_yearly,)),
        }
        if initial_params is not None:
            defaults.update(initial_params)

        # The compiled optimizer's layout is [k, m, delta(S), beta(K), zeta],
        # with zeta = log(sigma_obs) last -- see cpp_to_canonical. Estimating
        # the log keeps sigma_obs positive without box constraints, which
        # liblbfgs does not have.
        zeta_init = np.log(defaults.get('sigma_obs', SIGMA_OBS_INIT))
        params = np.concatenate(([defaults['k']], [defaults['m']],
                                 defaults['delta'], defaults['beta'], [zeta_init]))

        cpp = load_cpp_module(lib_path)
        result = cpp.optimize(
            params=params,
            t_scaled=self.t_scaled,
            change_points=self.change_points,
            t_seasonality=self.t_seasonality,
            normalized_y=self.normalized_y,
            sigma_obs_prior_scale=SIGMA_OBS_PRIOR_SCALE,
            sigma_k=self.sigma_k,
            sigma_m=self.sigma_m,
            sigma=self.sigma,
            tau=self.tau,
            verbose=verbose,
        )

        # Mirrors fit(), so both paths expose a comparable loss trajectory --
        # with one caveat: fit()'s is per iteration, while the C++ core's is a
        # monotone envelope over objective evaluations, since LBFGSpp offers no
        # per-iteration hook. Both decrease monotonically to the same value.
        self.loss_over_iterations = list(result.loss_trace)
        self.opt = result
        self.opt_status = result.status
        self.opt_status_message = result.status_message

        if not np.all(np.isfinite(result.params)):
            raise RuntimeError(
                f"the C++ optimizer returned non-finite parameters "
                f"(status {result.status}: {result.status_message})")

        # Back to the canonical (k, m, delta, sigma_obs, beta) layout, so
        # predict()/trend_forecast_uncertainty() work the same regardless of
        # which fit method produced opt_params.
        self.opt_params = cpp_to_canonical(result.params)
        self.sigma_obs = self.opt_params[SIGMA_OBS_IDX]

        # Return whatever values are necessary
        return -1
        
        
    def add_regressor(self, regressor: pd.Series) -> None:
        pass
    

    def make_future_dataframe(self, periods, include_history=True):
        last_date = pd.to_datetime(self.ds.max())  # Ensure last_date is a datetime object
        future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, periods + 1)]
        future_dates_df = pd.DataFrame(future_dates, columns=['ds'])

        if include_history:
            history_dates_df = pd.DataFrame(self.ds, columns=['ds'])
            future_df = pd.concat([history_dates_df, future_dates_df], ignore_index=True)
        else:
            future_df = future_dates_df

        return future_df
    
    def trend_forecast_uncertainty(self, horizon=30, n_samples=500):
        k, m, delta, _sigma_obs, beta = extract_params(self.opt_params)
        probability_changepoint = self.n_changepoints / self.T
        future_df = self.make_future_dataframe(horizon)
        
        # Normalize the future dates
        future_t_scaled = np.array((pd.to_datetime(future_df['ds']) - self.ds.min()) / (self.ds.max() - self.ds.min()))
        
        forecast = []
        lambda_mle = abs(delta).mean()  # MLE of laplace distribution's scale parameter
        
        for _ in range(n_samples):
            sample = np.random.random(future_t_scaled.shape)
            new_changepoints = future_t_scaled[sample <= probability_changepoint]
            
            new_delta = np.r_[delta, self.rng.laplace(0, lambda_mle, new_changepoints.shape[0])]
            new_change_points = np.r_[self.change_points, new_changepoints]
            future_trend = compute_trend(k, m, new_delta, new_change_points, future_t_scaled, self.y_absmax)
            future_trend = future_trend[:horizon]  # Ensure only the required horizon is included
            
            forecast.append(future_trend)
            
        forecast = np.array(forecast)
        quantiles = np.percentile(forecast, [2.5, 97.5], axis=0)
        
        return future_df, quantiles
    
    def predict(self, future_df):
        # Extract optimal parameters
        k, m, delta, _sigma_obs, beta = extract_params(self.opt_params)
        
        # Normalize future dates
        future_df['t_scaled'] = (pd.to_datetime(future_df['ds']) - self.ds.min()) / (self.ds.max() - self.ds.min())
        
        # Trend component calculation
        trend = compute_trend(k, m, delta, self.change_points, future_df['t_scaled'].values, self.y_absmax)

        # Seasonality component calculation
        x = fourier_components(seasonal_time(future_df['ds']), YEARLY_PERIOD, n_yearly)
        seasonality = x.dot(beta)

        # Combine trend and seasonality for the forecast
        yhat = trend + seasonality * self.y_absmax  # De-normalize the forecasted values

        # Create forecast DataFrame
        forecast = future_df[['ds']].copy()
        forecast['trend'] = trend
        
        # Add uncertainty intervals for trend
        _, quantiles = self.trend_forecast_uncertainty(horizon=len(future_df))
        forecast['trend_lower'] = quantiles[0, :]
        forecast['trend_upper'] = quantiles[1, :]
        
        # Now that 'trend_lower' and 'trend_upper' are defined, calculate 'yhat_lower' and 'yhat_upper'
        forecast['yhat_lower'] = forecast['trend_lower'] + seasonality * self.y_absmax
        forecast['yhat_upper'] = forecast['trend_upper'] + seasonality * self.y_absmax
        
        forecast['seasonality'] = seasonality * self.y_absmax
        
        forecast['yhat'] = yhat
        
        return forecast