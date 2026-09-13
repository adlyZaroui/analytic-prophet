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

        
    def _minus_log_posterior(self, params: np.array) -> float:
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
                      np.sum(beta**2) / (2*self.sigma**2) + \
                      np.sum(np.abs(delta)) / self.tau

        return minus_log_posterior

    def _gradient(self, params: np.array) -> np.array:
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
        ddelta = -np.sum(r[:, None] * (self.t_scaled[:, None] - self.change_points) * A, axis=0) / sigma_obs**2 + np.sign(delta) / self.tau
        dsigma_obs = np.array([self.T / sigma_obs - np.sum(r**2) / sigma_obs**3 + sigma_obs / SIGMA_OBS_PRIOR_SCALE**2])
        dbeta = -np.dot(r, x) / sigma_obs**2 + beta / self.sigma**2

        gradient = np.concatenate([dk, dm, ddelta, dsigma_obs, dbeta])

        return gradient

    def _minus_log_posteriorAndGradient(self, params: np.array) -> Tuple[float, np.array]:
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
                      np.sum(beta**2) / (2*self.sigma**2) + \
                      np.sum(np.abs(delta)) / self.tau

        dk = np.array([-np.sum(r * self.t_scaled) / sigma_obs**2 + k / self.sigma_k**2])
        dm = np.array([-np.sum(r) / sigma_obs**2 + m / self.sigma_m**2])
        ddelta = -np.sum(r[:, None] * (self.t_scaled[:, None] - self.change_points) * A, axis=0) / sigma_obs**2 + np.sign(delta) / self.tau
        dsigma_obs = np.array([self.T / sigma_obs - np.sum(r**2) / sigma_obs**3 + sigma_obs / SIGMA_OBS_PRIOR_SCALE**2])
        dbeta = -np.dot(r, x) / sigma_obs**2 + beta / self.sigma**2

        gradient = np.concatenate([dk, dm, ddelta, dsigma_obs, dbeta])

        return minus_log_posterior, gradient
        
    def fit(self, df: pd.DataFrame, analytic: bool=False, use_combined: bool=False, optimizer: str='L-BFGS-B') -> Tuple[float, float, np.array, np.array]:
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

        loss_over_iterations = []

        def callback(x):
            fobj = self._minus_log_posterior(x)
            loss_over_iterations.append(fobj)

        initial_params_array = from_dict_to_array(initial_params_dict)

        # sigma_obs must stay positive, mirroring Stan's `real<lower=0> sigma_obs`
        bounds = [(None, None)] * SIGMA_OBS_IDX + [(1e-6, None)] + \
                 [(None, None)] * (len(initial_params_array) - SIGMA_OBS_IDX - 1)

        if use_combined:
            opt_params = minimize(self._minus_log_posteriorAndGradient,
                    initial_params_array,
                    method=optimizer,
                    bounds=bounds,
                    options={'maxiter': 10000},
                    callback=callback,
                    jac=True)
        elif analytic:
            opt_params = minimize(self._minus_log_posterior,
                    initial_params_array,
                    method=optimizer,
                    bounds=bounds,
                    options={'maxiter': 10000},
                    callback=callback,
                    jac=lambda x: self._gradient(x))
        else:
            opt_params = minimize(self._minus_log_posterior,
                            initial_params_array,
                            method=optimizer,
                            bounds=bounds,
                            options={'maxiter': 10000},
                            callback=callback)
        self.opt = opt_params
        self.opt_params = opt_params.x
        self.sigma_obs = opt_params.x[SIGMA_OBS_IDX]
        self.loss_over_iterations = loss_over_iterations
    
    def fit_cpp(self, df: pd.DataFrame) -> Tuple[float, float, np.array, np.array]:
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
        # 0 + init_r * N(0, 1) - STAN initialization
        init_r = 2.0
        k_init = init_r * self.rng.normal()
        m_init = init_r * self.rng.normal()
        delta_init = self.rng.normal(loc=0.0, scale=init_r, size=(25,))
        beta_init = self.rng.normal(loc=0.0, scale=init_r, size=(2 * n_yearly,))

        # The compiled optimizer's own extract_params expects (k, m, delta, beta)
        # with no sigma_obs slot -- it does not estimate sigma_obs, which stays
        # fixed at self.sigma_obs and is passed to it separately below.
        params = np.concatenate(([k_init], [m_init], delta_init, beta_init))

        # Load the shared library
        lib = ctypes.CDLL('./liboptimization.so')

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
                         ctypes.c_double]
        
        lib.optimize.restype = None
        
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
             self.tau)

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