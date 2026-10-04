"""The optimizers this project runs, and the tolerances they stop on.

No Prophet counterpart: Stan supplies both there. `projected_newton` is the
Newton branch of Prophet's algorithm rule (#25), and the C++ core's twin
lives in optimize.cpp.
"""
import numpy as np
from scipy.optimize import OptimizeResult


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

STAN_TOL_OBJ = 1e-12               # tol_obj

STAN_TOL_REL_OBJ = 1e+4            # tol_rel_obj, scaled by STAN_EPS

STAN_TOL_GRAD = 1e-8               # tol_grad

STAN_TOL_PARAM = 1e-8              # tol_param

STAN_MAX_ITERATIONS = 10000        # Prophet passes iter=int(1e4)

# fit()'s one remaining deviation from those settings (#24), named so it can be
# referred to rather than buried as a literal. Stan's relative-objective
# threshold is STAN_TOL_REL_OBJ * STAN_EPS = 2.22e-12; scipy's iterate sequence
# on this objective takes single steps well below that while still nats from
# the optimum, and adopting the number would put this path *below* Prophet on
# the full series. The long form of the argument is in fit(), with the numbers.
SCIPY_TOL_REL_OBJ = 1e-16

# [fc] CmdStanPyBackend.fit: `'Newton' if T < 100 else 'LBFGS'`, with one retry
# on Newton when the first attempt raises. Strictly fewer than 100.
NEWTON_BELOW = 100

# [fc] Prophet.fit short-circuits a series that never moves: when `y.min() ==
# y.max()` under linear or flat growth it keeps the initial parameters, sets
# this as sigma_obs, and never calls the optimizer at all.
#
# The number is Prophet's literal, not a bound. fit()'s own lower bound on
# sigma_obs is 1e-6, which is exactly where a constant series lands when it is
# optimized the long way: the likelihood has no optimum in the interior, since
# the residuals are identically zero and -T*log(sigma) rises without limit as
# sigma falls, so the fit runs to whichever floor it is given. Prophet picks
# the floor up front and saves the iterations (#102).
CONSTANT_SERIES_SIGMA_OBS = 1e-9

# Which terminal status means "Stan would have raised here", and so triggers the
# Newton retry. The two solvers number theirs differently, and the numbering
# collides, so each path names its own rather than sharing a test: scipy's 2 is
# ABNORMAL_TERMINATION_IN_LNSRCH while its 1 is the iteration cap; the C++
# core's 1 is an exception out of LBFGSpp while its 2 is the cap. Only the
# first of each pair is a failure. Running out of `iter` is not one: CmdStan
# returns the result with a warning, and cmdstanpy raises only on a non-zero
# exit code, so Prophet keeps that fit rather than retrying it.
SCIPY_LINE_SEARCH_FAILURE = 2

CPP_SOLVER_RAISED = 1

def finite_difference_hessian(gradient_fn, x):
    """Central differences of an *analytic* gradient, symmetrized.

    2n gradient evaluations, accurate to about 1e-8 -- an order better than
    differencing the objective twice, and the reason a Newton step is available
    here at all without autodiff. Stan gets its Hessian from autodiff; this
    project's claim is about the gradient, and differencing an exact gradient
    is a different thing from differencing an objective.

    Mirrors finite_difference_hessian() in optimize.cpp, so the two Newton
    paths take the same steps.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    hessian = np.empty((n, n))
    point = x.copy()
    for i in range(n):
        step = np.sqrt(STAN_EPS) * max(1.0, abs(x[i]))
        point[i] = x[i] + step
        forward = gradient_fn(point)
        point[i] = x[i] - step
        backward = gradient_fn(point)
        point[i] = x[i]
        hessian[:, i] = (forward - backward) / (2.0 * step)
    # the two triangles differ by the differencing error alone
    return 0.5 * (hessian + hessian.T)

def projected_newton(objective, gradient_fn, z0, lower, upper):
    """Bound-constrained Newton, as the Newton branch of Prophet's rule.

    Prophet runs Stan's Newton below 100 observations and L-BFGS at or above
    ([fc] CmdStanPyBackend.fit), so this path needs a Newton of its own. It is
    the Python twin of newton() in optimize.cpp -- same split space, same
    Levenberg damping, same stopping tests -- so that the two fit paths stay
    comparable under the rule the way they are under L-BFGS.

    It runs on the split reformulation for the same reason L-BFGS-B does. On
    the natural parameterization the Laplace prior leaves a kink exactly where
    the optimum sits, and Newton has no mechanism to land on one: measured, it
    oscillates across the kink making about 1e-5 progress a step, still ~66
    nats short after Prophet's whole 10,000-iteration budget. Split, the L1
    term is linear, so its curvature is zero rather than undefined, and what is
    left is a smooth problem with bounds.

    Active set: a coordinate sitting at a bound while the gradient pushes it
    further into that bound is already as good as it gets. It does not count
    toward stationarity and is held out of the Newton system, which also keeps
    the system non-singular when a whole block of delta is pinned at zero.
    """
    z = np.clip(np.asarray(z0, dtype=float), lower, upper)
    n = z.size
    value = objective(z)
    gradient = gradient_fn(z)
    loss_trace = [value]

    damping = 1e-6
    iteration = 0
    message = "converged"
    status = 0

    while iteration < STAN_MAX_ITERATIONS:
        pinned = ((z <= lower) & (gradient > 0)) | ((z >= upper) & (gradient < 0))
        projected = np.where(pinned, 0.0, gradient)
        if np.max(np.abs(projected)) < STAN_TOL_GRAD:
            break

        free = np.flatnonzero(~pinned)
        hessian = finite_difference_hessian(gradient_fn, z)[np.ix_(free, free)]
        free_gradient = gradient[free]

        trial_point, trial_value = z, value
        improved = False
        alpha = 1.0
        for _ in range(40):
            damped = hessian + damping * np.eye(free.size)
            try:
                free_step = np.linalg.solve(damped, -free_gradient)
            except np.linalg.LinAlgError:
                free_step = None
            if free_step is not None and np.all(np.isfinite(free_step)):
                step = np.zeros(n)
                step[free] = free_step
                alpha = 1.0
                for _ in range(30):
                    trial_point = np.clip(z + alpha * step, lower, upper)
                    trial_value = objective(trial_point)
                    if np.isfinite(trial_value) and trial_value < value:
                        improved = True
                        break
                    alpha *= 0.5
            if improved:
                break
            damping *= 10.0
        if not improved:
            message = "no further progress"
            break
        # Levenberg, coupled to the line search rather than to success alone: a
        # step that had to be backtracked is a step the quadratic model was
        # trusted too far on, so the damping goes up even though the step was
        # accepted. Without this the full step is rejected almost every
        # iteration -- measured at T=50, a median of 14 backtracks, alpha 6e-5 --
        # and the run crawls: 658 iterations instead of 44, and on one series it
        # stops 0.05 nats short of where it otherwise lands.
        damping = damping * 10.0 if alpha < 1.0 else max(damping * 0.2, 1e-14)
        trial_gradient = gradient_fn(trial_point)

        # [fc] Stan stops as soon as ANY of its tests holds.
        scale = max(abs(value), abs(trial_value), 1.0)
        objective_change = value - trial_value
        parameter_change = np.max(np.abs(trial_point - z))
        converged = (objective_change < STAN_TOL_OBJ
                     or objective_change / scale < STAN_TOL_REL_OBJ * STAN_EPS
                     or parameter_change < STAN_TOL_PARAM)

        z, value, gradient = trial_point, trial_value, trial_gradient
        loss_trace.append(value)
        iteration += 1
        if converged:
            break
    else:
        status = -2
        message = "reached the iteration cap without converging"

    return OptimizeResult(x=z, fun=value, jac=gradient, nit=iteration,
                          status=status, success=status == 0, message=message,
                          loss_trace=loss_trace)
