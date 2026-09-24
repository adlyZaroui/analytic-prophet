"""Model constants, checked against facebook/prophet.

Split out of forecaster.py so the numbers the model is *defined* by are in
one place rather than interleaved with the code that uses them. The three
categories below are the reason this file exists at all: a scale is not a
value, and conflating them is how a prior silently becomes a parameter.
"""


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

# FIXED -- Fourier order N for yearly seasonality, giving 2N = 20 columns.
# [fc] set_auto_seasonalities, yearly fourier_order=10
# K is no longer this: a model's seasonality registry decides it (#16 task 2),
# and this only sets what the default registry puts in the yearly slot.
n_yearly = 10

# SCALE on delta, the changepoint rate adjustments: delta ~ double_exponential(0, changepoint_prior_scale).
# [stan] model block; [fc] Prophet.__init__ changepoint_prior_scale=0.05 (user-configurable)
TAU = 0.05

# SCALE on beta, the seasonality coefficients: beta ~ normal(0, sigmas).
# [stan] model block; [fc] Prophet.__init__ seasonality_prior_scale=10.0 (user-configurable)
# This is the model-wide default a seasonality inherits when it carries no
# prior_scale of its own. The vector Stan actually receives is built per column
# by seasonality_prior_scales() -- `sigmas` is vector[K] there, so seasonality,
# holiday and regressor terms can each have their own.
SIGMA = 10.0

# SCALE on sigma_obs, the observation noise: sigma_obs ~ normal(0, 0.5),
# truncated at 0 by `real<lower=0>`. [stan] parameters + model blocks.
# Hardcoded in prophet.stan -- not user-configurable.
SIGMA_OBS_PRIOR_SCALE = 0.5

# SCALE on k: k ~ normal(0, 5). [stan] model block. Hardcoded, not configurable.
sigma_k = 5

# SCALE on m: m ~ normal(0, 5). [stan] model block. Hardcoded, not configurable.
sigma_m = 5

# INIT -- Prophet overrides Stan's random init with explicit values, so Stan's
# `init_r * N(0, 1)` default is never reached. [fc] calculate_initial_params
# returns sigma_obs=1.0, delta=zeros(S), beta=zeros(K), and k/m from
# linear_growth_init (see below).
SIGMA_OBS_INIT = 1.0
