import torch
import numpy as np

from typing import Optional, Tuple, Callable
from abc import ABC, abstractmethod, abstractproperty
from gym.spaces import Box

# gravitation acceleration
grav = 9.80665


class Env(ABC):

    def __init__(
            self,
            device: torch.device,
            dt: float = 0.01,
            params: Optional[dict] = None
    ):
        super(Env, self).__init__()
        self.device = device

        # validate parameters, raise error if they're not valid
        if params is not None and not self.validate_params(params):
            raise ValueError(f"Parameters not valid: {params}")

        if params is None:
            self.params = self.default_param()
        else:
            self.params = params

        # make sure the time step is valid
        assert dt > 0.0
        self.dt = dt

        self._state = None
        self._action = None

        self._t = 0  # current simulation time

        # mode
        self._mode = 'test'

    @abstractmethod
    def reset(self) -> torch.Tensor:
        """
        Reset the environment and return the current states.

        Returns
        -------
        states: torch.Tensor,
            current states
        """
        pass

    @abstractmethod
    def step(self, u: torch.Tensor) -> Tuple[torch.Tensor, float, bool, dict]:
        """
        Simulation the system for one step.

        Parameters
        ----------
        u: torch.Tensor,
            actions

        Returns
        -------
        next_state: torch.Tensor,
            state of the next time step
        reward: float,
            reward signal
        done: bool,
            if the simulation is ended or not
        info: dict,
            other useful information
        """
        pass

    @abstractmethod
    def render(self) -> np.ndarray:
        """
        Plot the environment at the current time step.

        Returns
        -------
        fig: np.ndarray,
            an array of the figure of the current environment
        """
        pass

    @abstractmethod
    def plot_states(self, states: torch.Tensor) -> np.ndarray:
        """
        Plot the curve of state changes given the states.

        Parameters
        ----------
        states: torch.Tensor,
            given states

        Returns
        -------
        fig: np.ndarray,
            an array of the figure of the changes of the given states
        """
        pass

    @abstractmethod
    def default_param(self) -> dict:
        """
        Get default parameters.

        Returns
        -------
        params: dict,
            a dict of default parameters
        """
        pass

    @abstractmethod
    def validate_params(self, params: dict) -> bool:
        """
        Check the parameters to see whether they are valid or not.

        Parameters
        ----------
        params: dict,
            given parameters

        Returns
        -------
        valid: bool,
            whether the given parameters are valid or not
        """
        pass

    @abstractmethod
    def goal_mask(self, x: torch.Tensor) -> tuple:
        """
        Mask of the state to show if the state is in the goal region of each subsystem.

        Parameters
        ----------
        x: torch.Tensor,
            bs x sum(self.n_dims) tensor of state

        Returns
        -------
        goal_mask: tuple,
            self.n_systems tuple of bs tensor, if the state is in the goal region of each subsystem
        """
        pass

    @abstractmethod
    def sample_goal(self, batch_size: int) -> torch.Tensor:
        """
        Sample the goal states.

        Parameters
        ----------
        batch_size: int,
            number of states needed

        Returns
        -------
        goal: torch.Tensor,
            a batch of goal states
        """
        pass

    @abstractproperty
    def share_lyapunov(self) -> tuple:
        """A tuple of tensors where each tensor is a list of systems that share the same ISS Lyapunov function."""
        pass

    @abstractproperty
    def share_ctrl(self) -> tuple:
        """A tuple of tensors where each tensor is a list of systems that share the same controller."""
        pass

    @abstractproperty
    def n_systems(self) -> int:
        """Number of systems."""
        pass

    @abstractproperty
    def n_dims(self) -> tuple:
        """State dimensions of each subsystem."""
        pass

    @abstractproperty
    def n_controls(self) -> tuple:
        """Control dimensions of each subsystem."""
        pass

    @abstractproperty
    def max_episode_steps(self) -> int:
        """Maximum simulation time step of one episode."""
        pass

    @abstractproperty
    def state_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a tuple (upper, lower) describing the expected range of states for this system."""
        pass

    @abstractproperty
    def control_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a tuple (upper, lower) describing the range of allowable control limits for this system."""
        pass

    @abstractproperty
    def reward_limits(self) -> Tuple[float, float]:
        """Return a tuple (upper, lower) describing the range of possible reward for this system."""
        pass

    @abstractmethod
    def closed_loop_dynamics(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Return the state derivatives at state x and control input u
            dx/dt = f(x, u)

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
        pass

    @abstractmethod
    def u_nominal(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the nominal control for the nominal parameters.

        Parameters
        ----------
        x: torch.Tensor,
            bs x sum(self.n_dims) tensor of state

        Returns
        -------
        u_nominal: torch.Tensor,
            bs x sum(self.n_controls) tensor of controls
        """
        pass

    def train(self):
        """Change the mode of the environment to 'train'."""
        self._mode = 'train'

    def test(self):
        """Change the mode of the environment to 'test'."""
        self._mode = 'test'

    @property
    def mode(self) -> str:
        """Get the current mode of the environment."""
        return self._mode

    @property
    def observation_space(self) -> Box:
        """Same as Gym.observation_space."""
        return Box(
            low=self.state_limits[0].cpu().detach().numpy(),
            high=self.state_limits[1].cpu().detach().numpy()
        )

    @property
    def action_space(self) -> Box:
        """Same as Gym.action_space."""
        return Box(
            low=self.control_limits[0].cpu().detach().numpy(),
            high=self.control_limits[1].cpu().detach().numpy()
        )

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Simulate the environment for one time step.

        Parameters
        ----------
        x: torch.Tensor,
            current state
        u: torch.Tensor,
            given control input

        Returns
        -------
        x_next: torch.Tensor,
            next state
        """
        x_dot = self.closed_loop_dynamics(x, u)
        return x + (x_dot * self.dt)

    def sample_states(self, batch_size: int) -> torch.Tensor:
        """
        Sample states from the env.

        Parameters
        ----------
        batch_size: int,
            number of states needed

        Returns
        -------
        states: torch.Tensor,
            sampled states from the env
        """
        high, low = self.state_limits
        if torch.isinf(low).any() or torch.isinf(high).any():
            randn = torch.randn(batch_size, sum(self.n_dims), device=self.device)
            states = randn + self.goal_point
        else:
            rand = torch.rand(batch_size, sum(self.n_dims), device=self.device)
            states = rand * (high - low) + low
        return states

    @abstractproperty
    def goal_point(self) -> torch.Tensor:
        """
        Goal point of the system.

        Returns
        -------
        goal point: torch.Tensor,
            sum(self.n_dims) tensor of states
        """
        pass

    @property
    def state(self) -> torch.Tensor:
        """Get the current state."""
        if self._state is not None:
            return self._state.squeeze(0)
        else:
            raise ValueError('State is not initialized')

    @property
    def state_std(self) -> torch.Tensor:
        """Get the standard deviation of the states."""
        upper_limit, lower_limit = self.state_limits
        return 1 / np.sqrt(12) * (upper_limit - lower_limit)

    @property
    def metadata(self) -> dict:
        """Same as Gym.metadata."""
        return {}

    @property
    def u_eq(self):
        """Equilibrium point of the control input."""
        return torch.zeros(sum(self.n_controls), device=self.device)

    @property
    def reward_range(self) -> tuple:
        """Same as Gym.reward_range."""
        return self.reward_limits[1], self.reward_limits[0]

    def change_device(self, device: torch.device):
        """
        Change the device of the simulation.

        Parameters
        ----------
        device: torch.device,
            the target device (cpu or cuda)
        """
        self.device = device
