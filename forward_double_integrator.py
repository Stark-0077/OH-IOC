"""Double integrator of Section V-A: the OH-OCP (1) and the experiment parameters."""

from types import SimpleNamespace

import casadi as ca
import numpy as np

from oh_ocp import OptimalHorizonOCP


class DoubleIntegratorModel(OptimalHorizonOCP):
    """Double integrator with position and velocity states and an acceleration input.

    The parameters are the log weights of the control and terminal costs.
    """

    state_dimension = 2
    control_dimension = 1
    parameter_dimension = 2
    HORIZON_GUESS = 3.0  # initial guess of the horizon for IPOPT (s)

    @staticmethod
    def dynamics(x, u):
        return ca.vertcat(x[1], u[0])

    @staticmethod
    def running_cost(x, u, theta):
        return 1.0 + ca.exp(theta[0]) * u[0] ** 2

    @staticmethod
    def terminal_cost(x, theta):
        return ca.exp(theta[1]) * (x[0] ** 2 + x[1] ** 2)

    def set_initial_guess(self, x0):
        """Position falling linearly to zero over HORIZON_GUESS."""
        T = self.HORIZON_GUESS
        self.ocp.set_initial(self.x, ca.vertcat(x0[0] * (1.0 - self.ocp.t / T), -x0[0] / T))


# %% Parameter selection
SYSTEM = SimpleNamespace(
    name="Double integrator",  # label in the table
    model=DoubleIntegratorModel,
    x0=np.array([2.0, -1.0]),  # nominal initial state, moving towards the target
    theta_true=np.array([2.0, 2.0]),
    P0=0.01 * np.eye(2),
    seed=20261110,
)
