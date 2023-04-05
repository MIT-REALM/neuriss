import torch
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from typing import Tuple, Optional

from .affine import ControlAffineEnv


def _power_flow(delta: torch.Tensor, E: torch.Tensor, Y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate activate power P and reactivate power Q

    Parameters
    ----------
    delta: torch.Tensor,
        bs x n_systems
    E: torch.Tensor,
        bs x n_systems
    Y: torch.Tensor,
        n_systems x n_systems, nodal admittance matrix

    Returns
    -------
    P: torch.Tensor,
        bs x n_systems, active power
    Q: torch.Tensor,
        bs x n_systems, reactivate power
    """
    bs = delta.shape[0]
    n_systems = delta.shape[1]
    P = torch.zeros(bs, n_systems).type_as(delta)
    Q = torch.zeros(bs, n_systems).type_as(delta)
    if n_systems == 1:
        P = E ** 2 * torch.real(Y) + E * torch.abs(Y) * torch.cos(delta - torch.angle(-Y))
        Q = -E ** 2 * torch.imag(Y) + E * torch.abs(Y) * torch.sin(delta - torch.angle(-Y))
    else:
        for i in range(n_systems):
            P[:, i] = E[:, i] ** 2 * torch.real(Y[i, i])
            Q[:, i] = -E[:, i] ** 2 * torch.imag(Y[i, i])
            for k in range(n_systems):
                if k != i:
                    angle = delta[:, i] - delta[:, k] - torch.angle(Y[i, k])
                    P[:, i] = P[:, i] + E[:, i] * E[:, k] * torch.abs(Y[i, k]) * torch.cos(angle)
                    Q[:, i] = Q[:, i] + E[:, i] * E[:, k] * torch.abs(Y[i, k]) * torch.sin(angle)
    return P, Q


class Microgrid(ControlAffineEnv):
    """
    Microgrid system with angle droop control.

    States (for each subsystem):
        x[0] = delta
        x[1] = E

    Control inputs (for each subsystem):
        u[0] = u_delta
        u[1] = u_E

    The system is parameterized by (k is the id of the subsystem)
        n: number of subsystems
        M_a[k]: tracking time constant for delta
        M_v[k]: tracking time constant for E
        D_a[k]: droop gain for delta
        D_v[k]: droop gain for E
        Y[i, j]: complex, nodal admittance matrix
        delta_ref[k]: desired delta
        E_ref[k]: desired E
    """

    # state indices (same for each subsystem)
    DELTA = 0
    E = 1

    # control indices (same for each subsystem)
    U_DELTA = 0
    U_E = 1

    # max episode steps
    MAX_EPISODE_STEPS = 500

    def __init__(
            self,
            device: torch.device,
            dt: float = 0.01,
            params: Optional[dict] = None
    ):
        super(Microgrid, self).__init__(device, dt, params)

        # set up parameters
        self.N_SYSTEMS = self.params['n']
        self.N_DIMS = tuple([2 for _ in range(self.N_SYSTEMS)])
        self.N_CONTROLS = tuple([2 for _ in range(self.N_SYSTEMS)])

        # calculate equilibrium point of power
        self.P_ref, self.Q_ref = _power_flow(
            self.params['delta_ref'].unsqueeze(0), self.params['E_ref'].unsqueeze(0), self.params['Y'])

    def __repr__(self):
        return 'Microgrid'

    def _f(self, x: torch.Tensor) -> torch.Tensor:
        bs = x.shape[0]
        f = torch.zeros((bs, sum(self.n_dims), 1)).type_as(x)

        # extract states
        delta_dim = [Microgrid.DELTA + sum(self.n_dims[:i]) for i in range(self.n_systems)]
        delta = x[:, delta_dim]
        E_dim = [Microgrid.E + sum(self.n_dims[:i]) for i in range(self.n_systems)]
        E = x[:, E_dim]

        # calculate powers
        P, Q = _power_flow(delta, E, self.params['Y'])

        # calculate f matrix
        f[:, delta_dim, 0] = (1 / self.params['M_a']) * \
                             (-(delta - self.params['delta_ref']) + self.params['D_a'] * (self.P_ref - P))
        f[:, E_dim, 0] = (1 / self.params['M_v']) * \
                         (-(E - self.params['E_ref']) + self.params['D_v'] * (self.Q_ref - Q))

        return f

    def _g(self, x: torch.Tensor) -> torch.Tensor:
        bs = x.shape[0]
        g = torch.eye(sum(self.n_dims), sum(self.n_controls)).type_as(x).unsqueeze(0).repeat(bs, 1, 1)

        return g

    def reset(self) -> torch.Tensor:
        initial_conditions = torch.tensor([
            [-2., 2.],
            [-2., 2.],
        ], device=self.device).repeat(self.n_systems, 1)
        state = torch.rand(1, sum(self.n_dims), device=self.device)
        state = state * (initial_conditions[:, 1] - initial_conditions[:, 0]) + initial_conditions[:, 0] + \
                self.goal_point.unsqueeze(0)
        self._state = state
        self._t = 0
        return self.state

    def step(self, u: torch.Tensor) -> Tuple[torch.Tensor, float, bool, dict]:
        if u.ndim == 1:
            u = u.unsqueeze(0)

        # clamp given the control limits
        upper_u_lim, lower_u_lim = self.control_limits
        for dim_idx in range(sum(self.n_controls)):
            u[:, dim_idx] = torch.clamp(
                u[:, dim_idx],
                min=lower_u_lim[dim_idx].item(),
                max=upper_u_lim[dim_idx].item(),
            )

        # calculate returns
        self._state = self.forward(self._state, u)
        self._action = u
        self._t += 1
        upper_x_lim, lower_x_limit = self.state_limits
        done = self._t >= self.max_episode_steps or \
               (self._state > upper_x_lim).any() or (self._state < lower_x_limit).any()
        reward = torch.tensor(5., device=self.device)
        reward -= torch.norm(self.state - self.goal_point)
        reward = np.clip(reward.cpu().numpy(), self.reward_limits[1], self.reward_limits[0])

        return self.state, float(reward), done, {}

    def render(self) -> np.ndarray:
        fig, axs = plt.subplots(self.n_systems, figsize=(5, 5), dpi=500)
        canvas = FigureCanvas(fig)
        system_goal = torch.split(self.goal_point, list(self.n_dims), dim=0)
        for i, ax in enumerate(axs):
            x0 = system_goal[i].squeeze().cpu().numpy()
            ax.scatter(x0[0], x0[1], s=1, c='b')
        system_x = torch.split(self.state, list(self.n_dims), dim=0)
        for i, ax in enumerate(axs):
            ax.scatter(system_x[i][0].cpu().numpy(), system_x[i][1].cpu().numpy(), s=2, c='r')

        # range of the figure
        upper_limit, lower_limit = self.state_limits
        upper_limit = upper_limit.cpu().numpy()
        lower_limit = lower_limit.cpu().numpy()
        for i, ax in enumerate(axs):
            ax.set_xlim((lower_limit[2 * i], upper_limit[2 * i]))
            ax.set_ylim((lower_limit[2 * i + 1], upper_limit[2 * i + 1]))

        # generate numpy data
        canvas.draw()
        data = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()

        return data

    def plot_states(self, states: torch.Tensor) -> np.ndarray:
        # setup canvas
        fig, axs = plt.subplots(self.n_dims[0], figsize=(5, 5), dpi=500)
        canvas = FigureCanvas(fig)

        # extract states
        errors = states - self.goal_point
        delta_dim = [Microgrid.DELTA + sum(self.n_dims[:i]) for i in range(self.n_systems)]
        delta_errors = errors[:, delta_dim].cpu().detach().numpy()
        E_dim = [Microgrid.E + sum(self.n_dims[:i]) for i in range(self.n_systems)]
        E_errors = errors[:, E_dim].cpu().detach().numpy()
        time = np.asarray([i * self.dt for i in range(delta_errors.shape[0])])
        total_time = np.arange(0, self.max_episode_steps * self.dt, self.dt)
        upper_limit, lower_limit = self.state_limits
        lower_limit = lower_limit - self.goal_point
        upper_limit = upper_limit - self.goal_point

        # plot delta errors
        axs[0].plot(total_time, np.zeros_like(total_time), linestyle="-.")
        for i in range(delta_errors.shape[1]):
            axs[0].plot(time, delta_errors[:, i], label=rf"$\delta_{i}'$")
        axs[0].set_title(r'$\delta$ error')
        axs[0].set_xlabel('time (sec.)')
        axs[0].set_ylabel(r'$\delta$ error (rad.)')
        axs[0].legend(loc='upper right', bbox_to_anchor=(1.2, 1.05))
        axs[0].set_xlim((0., self.max_episode_steps * self.dt))
        axs[0].set_ylim((lower_limit[delta_dim].min().cpu().numpy(), upper_limit[delta_dim].max().cpu().numpy()))

        # plot E errors
        axs[1].plot(total_time, np.zeros_like(total_time), linestyle="-.")
        for i in range(E_errors.shape[1]):
            axs[1].plot(time, E_errors[:, i], label=rf"$E_{i}'$")
        axs[1].set_title(r'$E$ error')
        axs[1].set_xlabel('time (sec.)')
        axs[1].set_ylabel(r'$E$ error (p.u.)')
        axs[1].legend(loc='upper right', bbox_to_anchor=(1.2, 1.05))
        axs[1].set_xlim((0., self.max_episode_steps * self.dt))
        axs[1].set_ylim((lower_limit[E_dim].min().cpu().numpy(), upper_limit[E_dim].max().cpu().numpy()))

        # generate numpy data
        plt.tight_layout()
        canvas.draw()
        data = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()

        return data

    def default_param(self) -> dict:
        """
        Default params are adapted from IEEE 123-node Test Feeder without subsystem 5
        References:
            [1] Tong Huang, Sicun Gao, and Le Xie. A Neural Lyapunov Approach to Transient Stability
            Assessment of Power Electronic-interfaced Networked Microgrids. IEEE Transactions on
            Smart Grid 13.1 (2021): 106-118.
        """
        params = {'n': 4,
                  'M_a': torch.tensor([1.2, 1.0, 0.8, 1.0], device=self.device),
                  'M_v': torch.tensor([12, 10, 16, 10], device=self.device),
                  'D_a': torch.tensor([1.2, 1.2, 1.2, 1.2], device=self.device),
                  'D_v': torch.tensor([0.2, 0.2, 0.2, 0.2], device=self.device)}

        # calculate nodal admittance matrix
        Y12 = 1 / (1.2030 + 1j * 1.1034)
        Y13 = 1 / (1.0300 + 1j * 0.7400)
        Y34 = 1 / (1.5042 + 1j * 1.3554)
        params['Y'] = torch.tensor(
            [[Y12 + Y13, -Y12, -Y13, 0],
             [-Y12, Y12, 0, 0],
             [-Y13, 0, Y13 + Y34, -Y34],
             [0, 0, -Y34, Y34]], device=self.device
        )

        # equilibrium
        params['delta_ref'] = torch.tensor([0, -np.pi / 3, np.pi * 3 / 4, np.pi / 6], device=self.device)
        params['E_ref'] = torch.tensor([1., 1., 1., 1.], device=self.device)

        return params

    def validate_params(self, params: dict) -> bool:
        # check keys
        keys = params.keys()
        valid = 'n' in keys and 'M_a' in keys and 'M_v' in keys and 'D_a' in keys and 'D_v' in keys \
                and 'Y' in keys and 'delta_ref' in keys and 'E_ref' in keys

        # check dim
        dim = params['n']
        valid = valid and params['M_a'].ndim == 1 and params['M_a'].shape[0] == dim
        valid = valid and params['M_v'].ndim == 1 and params['M_v'].shape[0] == dim
        valid = valid and params['D_a'].ndim == 1 and params['D_a'].shape[0] == dim
        valid = valid and params['D_v'].ndim == 1 and params['D_v'].shape[0] == dim
        valid = valid and params['Y'].ndim == 2 and params['Y'].shape[0] == dim and params['Y'].shape[1] == dim
        valid = valid and params['delta_ref'].ndim == 1 and params['delta_ref'].shape[0] == dim
        valid = valid and params['E_ref'].ndim == 1 and params['E_ref'].shape[0] == dim

        return valid

    def goal_mask(self, x: torch.Tensor) -> tuple:
        system_x = torch.split(x, list(self.n_dims), dim=1)
        system_goal = torch.split(self.goal_point, list(self.n_dims))
        masks = []
        for i_system in range(self.n_systems):
            mask = torch.ones(system_x[i_system].shape[0]).type_as(x)
            dist = torch.norm((system_x[i_system] - system_goal[i_system]), dim=1)
            for j in range(dist.shape[0]):
                if dist[j] > 0.01:
                    mask[j] = 0
            masks.append(mask.bool())
        return tuple(masks)

    def sample_goal(self, batch_size: int) -> torch.Tensor:
        goal = torch.zeros(batch_size, sum(self.n_dims), device=self.device)
        for i_system in range(self.n_systems):
            goal[:, 2 * i_system] = self.params['delta_ref'][i_system]
            goal[:, 2 * i_system + 1] = self.params['E_ref'][i_system]
        return goal

    @property
    def share_lyapunov(self) -> tuple:
        return (
            torch.tensor([i for i in range(self.n_systems)], device=self.device),
        )

    @property
    def share_ctrl(self) -> tuple:
        return tuple([torch.tensor([i]) for i in range(self.n_systems)])

    @property
    def n_systems(self) -> int:
        return self.N_SYSTEMS

    @property
    def n_dims(self) -> tuple:
        return self.N_DIMS

    @property
    def n_controls(self) -> tuple:
        return self.N_CONTROLS

    @property
    def max_episode_steps(self) -> int:
        return Microgrid.MAX_EPISODE_STEPS

    @property
    def state_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        upper_limit = torch.ones(sum(self.n_dims), device=self.device, dtype=torch.float) * 3.
        lower_limit = -1.0 * upper_limit

        return upper_limit + self.goal_point, lower_limit + self.goal_point

    @property
    def control_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        upper_limit = torch.ones(sum(self.n_controls), device=self.device, dtype=torch.float) * 5.
        lower_limit = -1.0 * upper_limit

        return upper_limit, lower_limit

    @property
    def reward_limits(self) -> Tuple[float, float]:
        return 5.0, 0.0

    @property
    def goal_point(self) -> torch.Tensor:
        goal = torch.zeros(sum(self.n_dims), device=self.device)
        for i_system in range(self.n_systems):
            goal[2 * i_system] = self.params['delta_ref'][i_system]
            goal[2 * i_system + 1] = self.params['E_ref'][i_system]
        return goal

    def u_nominal(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        return torch.zeros(x.shape[0], sum(self.n_controls)).type_as(x)

    @property
    def use_linearized_controller(self) -> bool:
        return False
