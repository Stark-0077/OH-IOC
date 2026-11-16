"""The EKF (4) and its Jacobian, computed by Algorithm 1 of the paper.

Equation numbers refer to the manuscript.  The paper tells two sets of
matrices apart by typeface only: in ekf_update and run_ekf, P, Q and R are the
EKF matrices P_k, Q_k and R_k of (4); in sensitivity_equations, A, R, M, Q and
N are the calligraphic matrices of (12), and P, W and V those of (11) and (13c).
"""

import time
from functools import lru_cache
from types import SimpleNamespace

import casadi as ca
import numpy as np
from scipy.integrate import solve_ivp

EPSILON_T = 1e-3  # horizon margin eps_T of Algorithm 1 (s)
Q_SCALE = 1e-8  # Q_k = Q_SCALE * I, a tuning matrix (Section III-A)


def ekf_update(theta_hat, P, measurement, predicted, jacobian, Q, R):
    """One step of the EKF (4): returns the new estimate and Riccati matrix."""
    S = jacobian @ P @ jacobian.T + R  # the matrix inverted in (4a)
    gain = np.linalg.solve(S.T, (P @ jacobian.T).T).T  # (4a): P Xi' S^-1, without forming S^-1
    theta_hat = theta_hat + gain @ (measurement - predicted)  # (4b)
    Phi = np.eye(theta_hat.size) - gain @ jacobian
    P = Phi @ P @ Phi.T + gain @ R @ gain.T + Q  # (4c)
    return theta_hat, P


@lru_cache(maxsize=None)
def sensitivity_equations(model_class):
    """CasADi functions for Propositions 3.1 to 3.3, built once per model."""
    n = model_class.state_dimension
    m = model_class.control_dimension
    d = model_class.parameter_dimension
    x = ca.SX.sym("x", n)
    u = ca.SX.sym("u", m)
    theta = ca.SX.sym("theta", d)
    costate = ca.SX.sym("costate", n)
    P = ca.SX.sym("P", n, n)
    W = ca.SX.sym("W", n, d)
    V = ca.SX.sym("V", n)
    X_theta = ca.SX.sym("X_theta", n, d)
    X_T = ca.SX.sym("X_T", n)
    backward_state = ca.vertcat(costate, ca.vec(P), ca.vec(W), V)
    forward_state = ca.vertcat(ca.vec(X_theta), X_T)

    # Hamiltonian (8) and its derivatives.
    f = model_class.dynamics(x, u)
    H = model_class.running_cost(x, u, theta) + ca.dot(costate, f)
    H_x = ca.gradient(H, x)
    H_u = ca.gradient(H, u)
    H_theta = ca.gradient(H, theta)
    f_x, f_u = ca.jacobian(f, x), ca.jacobian(f, u)
    H_xx, H_xu = ca.jacobian(H_x, x), ca.jacobian(H_x, u)
    H_ux, H_uu = H_xu.T, ca.jacobian(H_u, u)
    H_utheta, H_xtheta = ca.jacobian(H_u, theta), ca.jacobian(H_x, theta)
    H_uu_inverse = ca.inv(H_uu)

    # Definitions (12).
    A = f_x - f_u @ H_uu_inverse @ H_ux
    R = f_u @ H_uu_inverse @ f_u.T
    M = -f_u @ H_uu_inverse @ H_utheta
    Q = H_xx - H_xu @ H_uu_inverse @ H_ux
    N = H_xtheta - H_xu @ H_uu_inverse @ H_utheta

    # Equations integrated backward from the terminal time.
    backward = ca.Function(
        "backward",
        [backward_state, x, u, theta],
        [
            ca.vertcat(
                -H_x,  # (9a)
                ca.vec(-(Q + A.T @ P + P @ A - P @ R @ P)),  # (11a)
                ca.vec((P @ R - A.T) @ W - P @ M - N),  # (11b)
                (P @ R - A.T) @ V,  # (13c)
            )
        ],
    )

    # Their terminal conditions.
    psi_x = ca.gradient(model_class.terminal_cost(x, theta), x)
    psi_xx, psi_xtheta = ca.jacobian(psi_x, x), ca.jacobian(psi_x, theta)

    def at_terminal(expression):
        """The expression with the costate at its terminal value psi_x of (9b)."""
        return ca.substitute(expression, costate, psi_x)

    V_terminal = psi_xx @ f + at_terminal(H_x)
    terminal = ca.Function(
        "terminal",
        [x, u, theta],
        [ca.vertcat(psi_x, ca.vec(psi_xx), ca.vec(psi_xtheta), V_terminal)],
    )

    # Equations integrated forward from time zero.
    forward = ca.Function(
        "forward",
        [forward_state, backward_state, x, u, theta],
        [
            ca.vertcat(
                ca.vec((A - R @ P) @ X_theta - R @ W + M),  # (10a)
                (A - R @ P) @ X_T - R @ V,  # (13a)
            )
        ],
    )

    # Sensitivities of the states and controls to the parameters and the horizon.
    coupling = H_ux + f_u.T @ P
    U_theta = -H_uu_inverse @ (coupling @ X_theta + f_u.T @ W + H_utheta)  # (10b)
    U_T = -H_uu_inverse @ (coupling @ X_T + f_u.T @ V)  # (13b)
    xi_sensitivities = ca.Function(
        "xi_sensitivities",
        [forward_state, backward_state, x, u, theta],
        [ca.vertcat(X_theta, U_theta), ca.vertcat(X_T, U_T)],
    )

    # Derivatives of the terminal Hamiltonian in Proposition 3.3.
    bracket = (at_terminal(H_x) + psi_xx @ f).T  # [H_x(T)' + H_lambda(T)' psi_xx(T)], H_lambda = f
    h_theta = bracket @ X_theta + f.T @ psi_xtheta + at_terminal(H_theta).T  # (22)
    h_T = bracket @ X_T + f.T @ V_terminal  # (23)
    h_derivatives = ca.Function("h_derivatives", [forward_state, x, u, theta], [h_theta.T, h_T])

    return SimpleNamespace(
        backward=backward,
        terminal=terminal,
        forward=forward,
        xi_sensitivities=xi_sensitivities,
        h_derivatives=h_derivatives,
    )


def flat(value):
    return np.asarray(value, dtype=float).reshape(-1)


def integrate(rhs, start, end, initial):
    """Integrate an ODE; returns the solution as a function of time and its final value."""
    result = solve_ivp(
        rhs, (start, end), initial, dense_output=True, method="DOP853", rtol=1e-10, atol=1e-12
    )
    if not result.success:
        raise RuntimeError(result.message)
    return result.sol, result.y[:, -1]


def sensitivities(model, theta, solution, sample_time):
    """Lines 2 and 3 of Algorithm 1 along a solution, with the costate of line 1.

    Returns the sensitivities of the states and controls at the sample time to
    the parameters and to the horizon (Propositions 3.1 and 3.2), and the
    derivatives (22) and (23) of Proposition 3.3.
    """
    equations = sensitivity_equations(type(model))
    n, d = model.state_dimension, model.parameter_dimension
    grid = solution.grid
    intervals = len(grid) - 1

    def state_and_control(t):
        xi = solution.xi(t)
        return xi[:n], xi[n:]

    # Line 2: the costate (9a), P (11a), W (11b) and V (13c), backward from T,
    # one control interval at a time.
    backward_path = [None] * intervals  # backward_path[i](t) for t in control interval i
    values = flat(equations.terminal(*state_and_control(grid[-1]), theta))
    for i in reversed(range(intervals)):
        backward_path[i], values = integrate(
            lambda t, y: flat(equations.backward(y, *state_and_control(t), theta)),
            grid[i + 1],
            grid[i],
            values,
        )

    # Line 3: X_theta (10a) and X_T (13a), forward from zero.
    forward_path = [None] * intervals
    values = np.zeros(n * d + n)
    for i in range(intervals):
        forward_path[i], values = integrate(
            lambda t, y: flat(equations.forward(y, backward_path[i](t), *state_and_control(t), theta)),
            grid[i],
            grid[i + 1],
            values,
        )
    X_at_T = values  # X_theta(T) and X_T(T)

    # U_theta (10b) and U_T (13b) at the sample time, in its control interval.
    i = min(max(int(np.searchsorted(grid, sample_time, side="right")) - 1, 0), intervals - 1)
    xi_theta, xi_T = equations.xi_sensitivities(
        forward_path[i](sample_time), backward_path[i](sample_time), *state_and_control(sample_time), theta
    )
    h_theta, h_T = equations.h_derivatives(X_at_T, *state_and_control(grid[-1]), theta)
    return np.asarray(xi_theta, dtype=float), flat(xi_T), flat(h_theta), float(h_T)


def measurement_jacobian(model, C, theta, solution, sample_time):
    """Jacobian Xi_k of (5) at one sample time, computed by Algorithm 1 via (7).

    Also returns the derivative (23), or NaN when the horizon bound is active.
    """
    xi_theta, xi_T, h_theta, h_T = sensitivities(model, theta, solution, sample_time)
    if solution.bound_active:  # lines 4-5: the horizon is fixed at t_k + eps_T
        return C @ xi_theta, np.nan  # generate_table.py counts the NaNs
    dT_dtheta = -h_theta / h_T  # line 7, (21)
    return C @ (xi_theta + np.outer(xi_T, dT_dtheta)), h_T  # line 8, (7)


def run_ekf(model, C, measurements, times, theta_initial, P0, noise_variance):
    """Run the EKF over a measurement sequence.

    Returns the estimates, starting with the initial one, the time taken by
    each update, and the derivative (23) used in each update.
    """
    theta_hat = np.array(theta_initial, dtype=float)
    P = np.array(P0, dtype=float)
    Q = Q_SCALE * np.eye(theta_hat.size)
    R = noise_variance * np.eye(len(C))
    estimates = [theta_hat]
    runtimes = []
    horizon_derivatives = []
    for sample_time, measurement in zip(times, measurements):
        start = time.perf_counter()
        # Solve the forward OH-OCP with theta_hat (Algorithm 1, line 1).
        solution = model.solve(theta_hat, minimum=sample_time + EPSILON_T)
        # Predicted measurement at the current sample time.
        predicted = C @ solution.xi(sample_time)
        # Jacobian (Algorithm 1, lines 2-8).
        jacobian, h_T = measurement_jacobian(model, C, theta_hat, solution, sample_time)
        # EKF update (4).
        theta_hat, P = ekf_update(theta_hat, P, measurement, predicted, jacobian, Q, R)
        # Store the runtime, the estimate and the derivative (23) for reporting and plotting.
        runtimes.append(time.perf_counter() - start)
        estimates.append(theta_hat)
        horizon_derivatives.append(h_T)
    return np.array(estimates), np.array(runtimes), np.array(horizon_derivatives)
