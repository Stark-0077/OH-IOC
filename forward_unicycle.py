"""Planar unicycle of Section V-B: the OH-OCP (1) and the experiment parameters."""

from types import SimpleNamespace

import casadi as ca
import numpy as np

from oh_ocp import OptimalHorizonOCP


class UnicycleModel(OptimalHorizonOCP):
    """Unicycle driving to the target pose at the origin.

    The states are the position and heading, and the inputs are the forward
    and angular velocities.  The parameters are the log weights of the two
    control costs and of the terminal position and heading costs.
    """

    state_dimension = 3
    control_dimension = 2
    parameter_dimension = 4
    HORIZON_GUESS = 3.5  # initial guess of the horizon for IPOPT (s)

    @staticmethod
    def dynamics(x, u):
        return ca.vertcat(u[0] * ca.cos(x[2]), u[0] * ca.sin(x[2]), u[1])

    @staticmethod
    def running_cost(x, u, theta):
        return 1.0 + ca.exp(theta[0]) * u[0] ** 2 + ca.exp(theta[1]) * u[1] ** 2

    @staticmethod
    def terminal_cost(x, theta):
        return ca.exp(theta[2]) * (x[0] ** 2 + x[1] ** 2) + ca.exp(theta[3]) * x[2] ** 2

    def set_initial_guess(self, x0):
        """State moving linearly to the origin over HORIZON_GUESS; unit speed, no turning."""
        self.ocp.set_initial(self.x, ca.DM(x0) * (1.0 - self.ocp.t / self.HORIZON_GUESS))
        self.ocp.set_initial(self.u, ca.vertcat(1.0, 0.0))


# %% Parameter selection
SYSTEM = SimpleNamespace(
    name="Unicycle",  # label in the table
    model=UnicycleModel,
    x0=np.array([-3.0, -2.0, 0.2]),  # nominal initial state
    theta_true=np.array([0.0, 0.0, 2.0, 2.0]),
    P0=0.01 * np.eye(4),
    seed=20271110,
)
