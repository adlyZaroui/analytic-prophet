import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import halfcauchy
from typing import Tuple
import ctypes # for the `fit_cpp` method

N_CHANGE_POINTS = 25 # number of change points - hyperparameter
TAU = 0.05 # changepoint prior scale - hyperparameter
SIGMA = 10 # seasonality prior scale - hyperparameter
SIGMA_OBS_PRIOR_SCALE = 0.5 # prior scale on sigma_obs, matches Prophet's `sigma_obs ~ normal(0, 0.5)`
SIGMA_OBS_INIT = 1.0 # MAP init value for sigma_obs, matches Prophet's stan_init

n_yearly = 10  # Number of Fourier terms for yearly seasonality
sigma_k = 5  # Prior scale for rate changes
sigma_m = 5  # Prior scale for rate offsets

# Parameter vector layout shared by the analytic posterior/gradient and by
# predict()/trend_forecast_uncertainty(): [k, m, delta (S), sigma_obs, beta (2*n_yearly)]
K_IDX = 0
M_IDX = 1
DELTA_SLICE = slice(2, 2 + N_CHANGE_POINTS)
SIGMA_OBS_IDX = DELTA_SLICE.stop
BETA_SLICE = slice(SIGMA_OBS_IDX + 1, SIGMA_OBS_IDX + 1 + 2 * n_yearly)

def det_dot(a, b):
    return (a * b[None, :]).sum(axis=-1)

def fourier_components(t_days, period, n):
    x = 2 * np.pi * np.arange(1, n + 1) / period
    x = x * t_days[:, None]
    x = np.concatenate((np.cos(x), np.sin(x)), axis=1)
    return x

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
    A = (t_scaled[:, None] > change_points) * 1
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
        self.changepoint_range = 0.8
        
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

        self.scale_period = None

    def get_parameters(self) -> np.array:
        return self.opt_params
        
    def _normalize_y(self) -> None:
        self.y_absmax = np.max(np.abs(self.y))
        self.normalized_y = np.array(self.y / self.y_absmax)
    
    def _generate_change_points(self) -> None:
        max_t_scaled = np.max(self.t_scaled)
        self.change_points = np.linspace(0, self.changepoint_range * max_t_scaled, self.n_changepoints + 1)[1:]

        
    def _minus_log_posterior(self, params: np.array, include_l1_prior: bool=True) -> float:
        k, m, delta, sigma_obs, beta = extract_params(params)

        # trend component
        A = (self.t_scaled[:, None] > self.change_points) * 1
        gamma = -self.change_points * delta
        g = (k + np.dot(A, delta)) * self.t_scaled + (m + np.dot(A, gamma))

        # seasonality component
        period = 365.25 / self.scale_period
        x = fourier_components(self.t_scaled, period, 10)
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

    def _gradient(self, params: np.array, include_l1_prior: bool=True) -> np.array:
        k, m, delta, sigma_obs, beta = extract_params(params)

        # trend component
        A = (self.t_scaled[:, None] > self.change_points) * 1
        gamma = -self.change_points * delta
        g = (k + np.dot(A, delta)) * self.t_scaled + (m + np.dot(A, gamma))

        # seasonality component
        period = 365.25 / self.scale_period
        x = fourier_components(self.t_scaled, period, 10)
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

    def _minus_log_posteriorAndGradient(self, params: np.array, include_l1_prior: bool=True) -> Tuple[float, np.array]:
        k, m, delta, sigma_obs, beta = extract_params(params)

        # trend component
        A = (self.t_scaled[:, None] > self.change_points) * 1
        gamma = -self.change_points * delta
        g = (k + np.dot(A, delta)) * self.t_scaled + (m + np.dot(A, gamma))

        # seasonality component
        period = 365.25 / self.scale_period
        x = fourier_components(self.t_scaled, period, 10)
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

    def _split_minus_log_posterior(self, z: np.array) -> float:
        smooth = self._minus_log_posterior(split_to_canonical(z), include_l1_prior=False)
        return smooth + self._split_l1_penalty(z)

    def _split_gradient(self, z: np.array) -> np.array:
        return self._canonical_gradient_to_split(
            self._gradient(split_to_canonical(z), include_l1_prior=False))

    def _split_minus_log_posteriorAndGradient(self, z: np.array) -> Tuple[float, np.array]:
        smooth, gradient = self._minus_log_posteriorAndGradient(
            split_to_canonical(z), include_l1_prior=False)
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
        self.scale_period = (self.ds.max() - self.ds.min()).days

        self._normalize_y()
        self._generate_change_points()

        initial_params_dict = {
            'k': 0,
            'm': 0,
            'delta': np.zeros((25,)),
            'sigma_obs': SIGMA_OBS_INIT,
            'beta': np.zeros((2 * n_yearly,))
        }
        if initial_params is not None:
            initial_params_dict.update(initial_params)
        if fixed_sigma_obs is not None:
            initial_params_dict['sigma_obs'] = fixed_sigma_obs

        loss_over_iterations = []

        def callback(z):
            fobj = self._minus_log_posterior(split_to_canonical(z))
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
            objective, jac = self._split_minus_log_posteriorAndGradient, True
        elif analytic:
            objective, jac = self._split_minus_log_posterior, self._split_gradient
        else:
            objective, jac = self._split_minus_log_posterior, None

        opt_params = minimize(objective,
                        z0,
                        method=optimizer,
                        bounds=bounds,
                        options={'maxiter': 10000},
                        callback=callback,
                        jac=jac)

        self.opt = opt_params
        self.opt_params = split_to_canonical(opt_params.x)
        self.sigma_obs = self.opt_params[SIGMA_OBS_IDX]
        self.loss_over_iterations = loss_over_iterations
    
    def fit_cpp(self, df: pd.DataFrame, initial_params: dict=None, lib_path: str=None, verbose: bool=False) -> Tuple[float, float, np.array, np.array]:
        self.y = df['y'].values

        if df['ds'].dtype != 'datetime64[ns]':
            self.ds = pd.to_datetime(df['ds'])
        else:
            self.ds = df['ds']

        self.t_scaled = np.array((self.ds - self.ds.min()) / (self.ds.max() - self.ds.min()))
        self.T = df.shape[0]

        self.scale_period = (self.ds.max() - self.ds.min()).days
        self._normalize_y()
        self._generate_change_points()

        # Initialize parameters
        # 0 + init_r * N(0, 1) - STAN initialization, unless overridden by
        # initial_params (e.g. to match fit()'s starting point for a parity test)
        init_r = 2.0
        defaults = {
            'k': init_r * self.rng.normal(),
            'm': init_r * self.rng.normal(),
            'delta': self.rng.normal(loc=0.0, scale=init_r, size=(25,)),
            'beta': self.rng.normal(loc=0.0, scale=init_r, size=(2 * n_yearly,)),
        }
        if initial_params is not None:
            defaults.update(initial_params)

        # The compiled optimizer's own extract_params expects (k, m, delta, beta)
        # with no sigma_obs slot -- it does not estimate sigma_obs, which stays
        # fixed at self.sigma_obs and is passed to it separately below.
        params = np.concatenate(([defaults['k']], [defaults['m']], defaults['delta'], defaults['beta']))

        # Load the shared library. Defaults to the compiled library sitting next
        # to this module; lib_path lets tests point at one built into a temp dir.
        if lib_path is None:
            lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'liboptimization.so')
        lib = ctypes.CDLL(lib_path)

        # Define argument and return types for the optimize function
        lib.optimize.argtypes = [np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
                         ctypes.c_int,
                         np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
                         ctypes.c_int,
                         np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
                         ctypes.c_int,
                         ctypes.c_double,
                         np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
                         ctypes.c_int,
                         ctypes.c_double,
                         ctypes.c_double,
                         ctypes.c_double,
                         ctypes.c_double,
                         ctypes.c_double,
                         np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
                         ctypes.c_int,
                         ctypes.POINTER(ctypes.c_int),
                         ctypes.POINTER(ctypes.c_int),
                         ctypes.c_int]

        lib.optimize.restype = None

        max_iterations = 10000
        loss_over_iterations = np.zeros(max_iterations)
        n_iterations = ctypes.c_int(0)
        status = ctypes.c_int(0)

        lib.optimize(params,
             len(params),
             self.t_scaled,
             len(self.t_scaled),
             self.change_points,
             len(self.change_points),
             self.scale_period,
             self.normalized_y,
             len(self.normalized_y),
             self.sigma_obs,
             self.sigma_k,
             self.sigma_m,
             self.sigma,
             self.tau,
             loss_over_iterations,
             max_iterations,
             ctypes.byref(n_iterations),
             ctypes.byref(status),
             int(verbose))

        # Mirrors fit(): the per-iteration objective, so both fit paths expose
        # a directly comparable loss trajectory.
        self.loss_over_iterations = list(loss_over_iterations[:n_iterations.value])
        self.opt_status = status.value

        # Splice the fixed sigma_obs into the canonical (k, m, delta, sigma_obs,
        # beta) layout so predict()/trend_forecast_uncertainty() work the same
        # regardless of which fit method produced opt_params.
        self.opt_params = np.concatenate((params[:SIGMA_OBS_IDX], [self.sigma_obs], params[SIGMA_OBS_IDX:]))

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
        x = fourier_components(self.t_scaled, 365.25, n_yearly)
        s = det_dot(x, beta)
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
        period = 365.25 / self.scale_period
        x = fourier_components(future_df['t_scaled'].values, period, n_yearly)
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