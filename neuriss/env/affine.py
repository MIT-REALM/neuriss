import torch
import numpy as np

from abc import ABC, abstractmethod
from typing import Optional, Tuple
from torch.autograd.functional import jacobian

from .base import Env
from .utils import lqr, continuous_lyap


class ControlAffineEnv(Env, ABC):

    def __init__(
            self,
            device: torch.device,
            dt: float = 0.01,
            params: Optional[dict] = None
    ):
        super(ControlAffineEnv, self).__init__(device, dt, params)

        self._K = None
        self._P = None

    @abstractmethod
    def _f(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the control-independent part of the control-affine dynamics.

        Parameters
        ----------
        x: torch.Tensor,
            batch_size x sum(self.n_dims) tensor of state

        Returns
        -------
        f: torch.Tensor,
            batch_size x sum(self.n_dims) x 1
        """
        pass

    @abstractmethod
    def _g(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the control-independent part of the control-affine dynamics.

        Parameters
        ----------
        x: torch.Tensor,
            batch_size x sum(self.n_dims) tensor of state

        Returns
        -------
        g: torch.Tensor,
            batch_size x sum(self.n_dims) x sum(self.n_controls)
        """
        pass

    def control_affine_dynamics(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (f, g) representing the system dynamics in control-affine form:
            dx/dt = f(x) + g(x) u

        Parameters
        ----------
        x: torch.Tensor,
            batch_size x sum(self.n_dims) tensor of state

        Returns
        -------
        f: torch.Tensor,
            batch_size x sum(self.n_dims) x 1 representing the control-independent dynamics
        g: torch.Tensor,
            batch_size x sum(self.n_dims) x sum(self.n_controls) representing the control-dependent dynamics
        """
        if x.ndim == 1:
            x = x.unsqueeze(0)

        # sanity check on input
        assert x.ndim == 2
        assert x.shape[1] == sum(self.n_dims)

        return self._f(x), self._g(x)

    def closed_loop_dynamics(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Return the state derivatives at state x and control input u
            dx/dt = f(x) + g(x) u

        Parameters
        ----------
        x: torch.Tensor,
            batch_size x sum(self.n_dims) tensor of state
        u: torch.Tensor,
            batch_size x sum(self.n_controls) tensor of controls

        Returns
        -------
        x_dot: torch.Tensor,
            batch_size x sum(self.n_dims) tensor of time derivatives of x
        """
        # get the control-affine dynamics
        f, g = self.control_affine_dynamics(x)

        # compute state derivatives using control-affine form
        x_dot = f + torch.bmm(g, u.unsqueeze(-1))
        return x_dot.view(x.shape)

    @torch.enable_grad()
    def compute_A_matrix(self) -> np.ndarray:
        """Compute the linearized continuous-time state-state derivative transfer matrix about the goal point."""
        # linearize the system about the x = 0, u = 0
        x0 = self.goal_point.unsqueeze(0)
        u0 = self.u_eq.unsqueeze(0)

        def dynamics(x):
            return self.closed_loop_dynamics(x, u0).squeeze()

        A = jacobian(dynamics, x0).squeeze().cpu().numpy()
        A = np.reshape(A, (sum(self.n_dims), sum(self.n_dims)))

        return A

    def compute_B_matrix(self) -> np.ndarray:
        """Compute the linearized continuous-time state-state derivative transfer matrix about the goal point."""

        # linearize the system about the x = 0, u = 0
        x0 = self.goal_point.unsqueeze(0)
        B = self._g(x0).squeeze().cpu().numpy()
        B = np.reshape(B, (sum(self.n_dims), sum(self.n_controls)))

        return B

    def linearized_ct_dynamics_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the continuous time linear dynamics matrices, dx/dt = Ax + Bu."""
        A = self.compute_A_matrix()
        B = self.compute_B_matrix()

        return A, B

    def linearized_dt_dynamics_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the continuous time linear dynamics matrices, x_{t+1} = Ax_{t} + Bu."""
        Act, Bct = self.linearized_ct_dynamics_matrices()
        A = np.eye(sum(self.n_dims)) + self.dt * Act
        B = self.dt * Bct

        return A, B

    def compute_linearized_controller(self):
        """Computes the linearized controller K and lyapunov matrix P."""
        # Compute the LQR gain matrix for the nominal parameters
        Act, Bct = self.linearized_ct_dynamics_matrices()
        A, B = self.linearized_dt_dynamics_matrices()

        # Define cost matrices as identity
        Q = np.eye(sum(self.n_dims))
        R = np.eye(sum(self.n_controls))

        # Get feedback matrix
        K_np = lqr(A, B, Q, R)
        self._K = torch.tensor(K_np, device=self.device)

        Acl = Act - Bct @ K_np

        # use the standard Lyapunov equation
        self._P = torch.tensor(continuous_lyap(Acl, Q), device=self.device)

    def u_nominal(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the nominal control for the nominal parameters, using LQR unless overridden.

        Parameters
        ----------
        x: torch.Tensor,
            bs x sum(self.n_dims) tensor of state

        Returns
        -------
        u_nominal: torch.Tensor,
            bs x sum(self.n_controls) tensor of controls
        """
        if self._K is None:
            raise KeyError('u_nominal is not computed, call compute_linearized_controller() first')

        # Compute nominal control from feedback + equilibrium control
        K = self._K.type_as(x)
        x0 = self.goal_point.unsqueeze(0)
        goal = x0.squeeze().type_as(x)
        u_nominal = -(K @ (x - goal).T).T

        # Adjust for the equilibrium setpoint
        u = u_nominal + self.u_eq.unsqueeze(0).type_as(x)

        # clamp given the control limits
        upper_u_lim, lower_u_lim = self.control_limits
        for dim_idx in range(sum(self.n_controls)):
            u[:, dim_idx] = torch.clamp(
                u[:, dim_idx],
                min=lower_u_lim[dim_idx].item(),
                max=upper_u_lim[dim_idx].item(),
            )

        return u

    @property
    def use_linearized_controller(self) -> bool:
        """Whether to use linearized controller as the nominal controller."""
        return True

    def save_nominal_controller(self, path: str):
        """
        Save the computed nominal controller.

        Parameters
        ----------
        path: str,
            path to save the nominal controller
        """
        torch.save(self._K, path)

    def load_nominal_controller(self, path: str):
        """
        Load the pre-computed nominal controller.

        Parameters
        ----------
        path: str,
            path to load the nominal controller
        """
        self._K = torch.load(path, map_location=self.device)
