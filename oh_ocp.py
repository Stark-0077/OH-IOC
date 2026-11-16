"""The optimal-horizon OCP (1), solved numerically with Rockit and IPOPT.

forward_double_integrator.py and forward_unicycle.py define the two systems of Section V as subclasses of OptimalHorizonOCP.
"""

import casadi as ca
import numpy as np
from rockit import FreeTime, MultipleShooting, Ocp

INTERVALS = 20  # control intervals of the multiple-shooting transcription
HORIZON_FLOOR = 0.05  # smallest horizon the solver may return (s)
BOUND_TOLERANCE = 1e-6  # tolerance for an active horizon bound
NODES = np.array([0.1, 0.3, 0.5, 0.7, 0.9])  # sample points within a control interval


class Solution:
    """Optimal horizon and trajectory from one solve of the OCP."""

    def __init__(self, horizon, bound_active, grid, coefficients):
        self.horizon = horizon
        self.bound_active = bound_active  # whether the lower bound on the horizon is active
        self.grid = grid  # boundaries of the control intervals
        self.coefficients = coefficients  # one polynomial per control interval

    def xi(self, time):
        """States and controls col(x, u) at a time within the horizon."""
        intervals = len(self.coefficients)
        width = self.horizon / intervals
        index = min(max(int(time / width), 0), intervals - 1)
        tau = time / width - index
        return tau ** np.arange(NODES.size) @ self.coefficients[index]


class OptimalHorizonOCP:
    """The OH-OCP (1) from one initial state x0.

    A subclass defines state_dimension, control_dimension and
    parameter_dimension; the static methods dynamics f(x, u),
    running_cost g(x, u, theta) and terminal_cost psi(x, theta);
    HORIZON_GUESS; and set_initial_guess(x0).
    """

    def __init__(self, x0):
        """Set up the OCP from the initial state x0."""
        self.ocp = Ocp(T=FreeTime(self.HORIZON_GUESS))
        self.x = self.ocp.state(self.state_dimension)
        self.u = self.ocp.control(self.control_dimension, order=1)  # piecewise linear
        self.theta = self.ocp.parameter(self.parameter_dimension)
        self.minimum = self.ocp.parameter()
        self.ocp.set_der(self.x, self.dynamics(self.x, self.u))
        self.ocp.subject_to(self.ocp.at_t0(self.x) == ca.DM(x0))
        self.ocp.subject_to(self.ocp.T >= self.minimum)
        self.ocp.add_objective(
            self.ocp.integral(self.running_cost(self.x, self.u, self.theta))
            + self.ocp.at_tf(self.terminal_cost(self.x, self.theta))
        )
        self.ocp.solver(
            "ipopt",
            {
                "expand": True,
                "print_time": False,
                "ipopt": {"print_level": 0, "sb": "yes", "tol": 1e-10, "max_iter": 250},
            },
        )
        self.ocp.method(MultipleShooting(N=INTERVALS, intg="rk"))  # one RK4 step per interval
        self.set_initial_guess(x0)

    def solve(self, theta, minimum=HORIZON_FLOOR):
        """Solve the OCP for theta with the horizon bounded below by minimum.

        IPOPT starts from the same initial guess every time, so this is the
        local optimum reached from that guess.
        """
        minimum = max(minimum, HORIZON_FLOOR)
        self.ocp.set_value(self.theta, theta)
        self.ocp.set_value(self.minimum, minimum)
        solution = self.ocp.solve()
        horizon = float(solution.value(self.ocp.T))
        bound_active = horizon <= minimum + BOUND_TOLERANCE * (1.0 + minimum)

        # Sample the trajectory in each control interval and fit a polynomial to samples so ODEs in ekf.py can be evaluated cheaply at any time.
        grid = np.linspace(0.0, horizon, INTERVALS + 1)
        times = (grid[:-1, None] + horizon / INTERVALS * NODES).ravel()
        states, controls = solution.sampler([self.x, self.u])(times)
        samples = np.hstack(
            [np.reshape(states, (times.size, -1)), np.reshape(controls, (times.size, -1))]
        )
        coefficients = np.linalg.solve(
            np.vander(NODES, increasing=True), samples.reshape(INTERVALS, NODES.size, -1)
        )
        return Solution(horizon, bound_active, grid, coefficients)
