"""Where each parameter sits in the flat vector, and how to move between
the three shapes it takes.

No Prophet counterpart: Stan generates this bookkeeping from its `parameters`
block, so the original never writes it down. Here it is explicit, which is
what lets S and K vary (#3) and what keeps the Python and C++ layouts from
drifting apart.
"""
import numpy as np

from .constants import N_CHANGE_POINTS, n_yearly


class ParameterLayout:
    """Where each block sits in the parameter vector, given S changepoints and
    K seasonality columns: [k, m, delta(S), sigma_obs, beta(K)].

    This used to be five module constants fixed at import, which pinned the
    model to 25 changepoints and one order-10 seasonality. The C++ core stopped
    assuming either in #3; carrying the layout as a value makes the Python side
    follow, which is what lets a seasonality registry exist at all.
    """

    __slots__ = ("n_changepoints", "n_seasonality_columns", "n_holiday_columns",
                 "k_idx", "m_idx", "delta", "sigma_obs_idx", "beta", "size")

    def __init__(self, n_changepoints, n_seasonality_columns, n_holiday_columns=0):
        self.n_changepoints = n_changepoints
        self.n_seasonality_columns = n_seasonality_columns
        self.n_holiday_columns = n_holiday_columns
        n_seasonality_columns += n_holiday_columns
        self.k_idx = 0
        self.m_idx = 1
        self.delta = slice(2, 2 + n_changepoints)
        self.sigma_obs_idx = 2 + n_changepoints
        self.beta = slice(3 + n_changepoints, 3 + n_changepoints + n_seasonality_columns)
        self.size = 3 + n_changepoints + n_seasonality_columns

    @property
    def n_regressor_columns(self):
        """K -- every column of the design matrix, seasonal and holiday alike.

        The objective does not distinguish them: `beta` spans the lot and the
        prior is per column either way. The split is kept only so the two
        blocks can be built and sliced separately.
        """
        return self.n_seasonality_columns + self.n_holiday_columns

    @property
    def holidays(self):
        """The holiday coefficients' slice of the *parameter vector*.

        Indexed like `beta` and `delta`, so it applies to the optimizer's
        flat vector. For the
        design matrix and `sigmas` -- which are indexed by column, from 0 -- use
        `holiday_block`. The two differ by the 3 + S offset, and mixing them
        silently returns the wrong slice rather than raising.
        """
        return slice(self.beta.start + self.n_seasonality_columns, self.beta.stop)

    @property
    def holiday_block(self):
        """The holiday columns' slice of the *design matrix* and of `sigmas`."""
        return slice(self.n_seasonality_columns, self.n_regressor_columns)

    @property
    def seasonality_block(self):
        """The seasonal columns' slice of the design matrix and of `sigmas`."""
        return slice(0, self.n_seasonality_columns)

    def __eq__(self, other):
        """A value, so two layouts over the same counts are the same layout.

        Every other field is derived from these three, so comparing them is
        comparing the whole object. It exists so that a refit and a fresh fit
        can be compared attribute by attribute (#41) without identity getting
        in the way.
        """
        if not isinstance(other, ParameterLayout):
            return NotImplemented
        return (self.n_changepoints, self.n_seasonality_columns,
                self.n_holiday_columns) == (other.n_changepoints,
                                            other.n_seasonality_columns,
                                            other.n_holiday_columns)

    def __hash__(self):
        return hash((self.n_changepoints, self.n_seasonality_columns,
                     self.n_holiday_columns))

    def __repr__(self):
        return (f"ParameterLayout(S={self.n_changepoints}, "
                f"K={self.n_regressor_columns} "
                f"({self.n_seasonality_columns} seasonal + "
                f"{self.n_holiday_columns} holiday), size={self.size})")

# The layout a default model uses: 25 changepoints, yearly seasonality at
# order 10. The module constants below are its fields, kept so that callers
# reading a default-shaped vector need no layout in hand.
DEFAULT_LAYOUT = ParameterLayout(N_CHANGE_POINTS, 2 * n_yearly)

K_IDX = DEFAULT_LAYOUT.k_idx

M_IDX = DEFAULT_LAYOUT.m_idx

DELTA_SLICE = DEFAULT_LAYOUT.delta

SIGMA_OBS_IDX = DEFAULT_LAYOUT.sigma_obs_idx

BETA_SLICE = DEFAULT_LAYOUT.beta

def extract_params(params, layout=DEFAULT_LAYOUT):
    k = params[layout.k_idx]
    m = params[layout.m_idx]
    delta = params[layout.delta]
    sigma_obs = params[layout.sigma_obs_idx]
    beta = params[layout.beta]
    return k, m, delta, sigma_obs, beta

def from_dict_to_array(params, layout=DEFAULT_LAYOUT):
    """Pack a parameter dict into a vector in `layout`'s order.

    `beta` used to be overwritten with zeros here regardless of what was
    passed (#34), which was invisible only because every caller happened to
    pass zeros.
    """
    beta = np.asarray(params['beta'], dtype=float)
    if beta.shape != (layout.n_regressor_columns,):
        raise ValueError(
            f"beta has {beta.shape} entries but the layout expects "
            f"{layout.n_regressor_columns}")
    return np.concatenate(([params['k']], [params['m']], np.asarray(params['delta'], dtype=float),
                           [params['sigma_obs']], beta))

def canonical_to_split(params, layout=DEFAULT_LAYOUT):
    """(k, m, delta, sigma_obs, beta) -> (k, m, delta_pos, delta_neg, sigma_obs, beta).

    The Laplace prior on delta puts |delta|/changepoint_prior_scale in the objective, which is not
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
    k, m, delta, sigma_obs, beta = extract_params(params, layout)
    return np.concatenate(([k], [m], np.maximum(delta, 0), np.maximum(-delta, 0), [sigma_obs], beta))

def split_to_canonical(z, n_delta=N_CHANGE_POINTS):
    """Inverse of canonical_to_split: delta = delta_pos - delta_neg."""
    k, m = z[0], z[1]
    delta_pos = z[2:2 + n_delta]
    delta_neg = z[2 + n_delta:2 + 2 * n_delta]
    sigma_obs = z[2 + 2 * n_delta]
    beta = z[3 + 2 * n_delta:]
    return np.concatenate(([k], [m], delta_pos - delta_neg, [sigma_obs], beta))
