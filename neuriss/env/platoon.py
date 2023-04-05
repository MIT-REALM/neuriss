import copy
import torch
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from typing import Tuple, Optional

from .affine import ControlAffineEnv
from .utils import plot_rectangle, lqr


class Platoon(ControlAffineEnv):
    """
    Platoon system
    """

    # state indices (same for each subsystem)
    X_FRONT = 0
    X_BACK = 1
    V = 2

    # control indices (same for each subsystem)
    F = 0

    # max episode steps
    MAX_EPISODE_STEPS = 500

    # valid paths
    VALID_SPEED_PROFILE = (
        'sin',
        'constant'
    )

    def __init__(
            self,
            device: torch.device,
            dt: float = 0.01,
            params: Optional[dict] = None
    ):
        super(Platoon, self).__init__(device, dt, params)

        # set up parameters
        self.N_SYSTEMS = self.params['n']
        self.N_DIMS = tuple([3 for _ in range(self.N_SYSTEMS)])
        self.N_CONTROLS = tuple([1 for _ in range(self.N_SYSTEMS)])

        # tracking data
        self._ref_path = self._generate_ref(self.params['speed profile'])
        self._info = None

    def __repr__(self):
        return 'Platoon'

    def _set_info(self, t: int) -> dict:
        info = {}
        while t >= self._ref_path.shape[0]:
            t -= self._ref_path.shape[0]
        info['x_ref'] = self._ref_path[t, 0]
        info['v_ref'] = self._ref_path[t, 1]
        return info

    def _generate_ref(self, speed_profile: str) -> torch.Tensor:
        ref_path = torch.zeros(self.max_episode_steps, 2, device=self.device)

        if speed_profile == 'constant':
            x_ref = 0.0
            v_ref = self.params['v_init']
            for step in range(self.max_episode_steps):
                x_ref += self.dt * v_ref
                ref_path[step, :].copy_(torch.tensor([x_ref, v_ref]))
        elif speed_profile == 'sin':
            x_ref = 0.0
            v_ref = self.params['v_init']
            for step in range(self.max_episode_steps):
                a_ref = 1 * np.sin(step * self.dt * 5)
                v_ref += self.dt * a_ref
                v_ref = np.clip(v_ref, a_min=0.5, a_max=4.0)
                x_ref += self.dt * v_ref
                ref_path[step, :].copy_(torch.tensor([x_ref, v_ref]))
        else:
            raise NotImplementedError('Dubins car error: Unknown path')

        return ref_path

    def _f(self, x: torch.Tensor) -> torch.Tensor:
        bs = x.shape[0]
        f = torch.zeros((bs, sum(self.n_dims), 1)).type_as(x)

        # extract states
        x_front_dim = [Platoon.X_FRONT + sum(self.n_dims[:i]) for i in range(self.n_systems)]
        x_back_dim = [Platoon.X_BACK + sum(self.n_dims[:i]) for i in range(self.n_systems)]
        v_dim = [Platoon.V + sum(self.n_dims[:i]) for i in range(self.n_systems)]
        v = x[:, v_dim]

        info = self._set_info(self._t)
        if self.mode == 'test':
            v_ref = info['v_ref']
        else:
            if np.random.rand() < 0.5:
                v_ref = info['v_ref'].max()
            else:
                v_ref = info['v_ref'].min()

        # calculate f matrix
        for i, dim in enumerate(x_front_dim):
            if i == 0:
                f[:, dim, 0] = v_ref - v[:, i]
            else:
                f[:, dim, 0] = v[:, i - 1] - v[:, i]
        for i, dim in enumerate(x_back_dim):
            if i == len(x_back_dim) - 1:
                f[:, dim, 0] = v[:, i] - v_ref
            else:
                f[:, dim, 0] = v[:, i] - v[:, i + 1]

        return f

    def _g(self, x: torch.Tensor) -> torch.Tensor:
        bs = x.shape[0]
        g = torch.zeros((bs, sum(self.n_dims), sum(self.n_controls))).type_as(x)

        # extract dims
        v_dim = [Platoon.V + sum(self.n_dims[:i]) for i in range(self.n_systems)]
        f_dim = [Platoon.F + sum(self.n_controls[:i]) for i in range(self.n_systems)]
        for i, v in enumerate(v_dim):
            g[:, v, f_dim[i]] = 1 / self.params['m'][i]

        return g

    def reset(self) -> torch.Tensor:
        self._t = 0
        self._ref_path = self._generate_ref(self.params["speed profile"])
        self._info = self._set_info(t=0)

        state = torch.zeros(1, sum(self.n_dims), device=self.device)
        x = [(i + 1) * self.params['r'] + np.random.rand() * self.params['r'] * 0.2 for i in range(self.n_systems)]
        for i_system in range(self.n_systems):
            if i_system == 0:
                state[0, Platoon.X_FRONT + sum(self.n_dims[:i_system])] = x[i_system]
            else:
                state[0, Platoon.X_FRONT + sum(self.n_dims[:i_system])] = x[i_system] - x[i_system - 1]
            if i_system == self.n_systems - 1:
                state[0, Platoon.X_BACK + sum(self.n_dims[:i_system])] = \
                    self.params['r'] * (self.n_systems + 1) - x[i_system]
            else:
                state[0, Platoon.X_BACK + sum(self.n_dims[:i_system])] = x[i_system + 1] - x[i_system]
            state[0, Platoon.V + sum(self.n_dims[:i_system])] = self.params['v_init'] * (0.5 + np.random.rand() * 0.1)

        self._state = state
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
        self._info = self._set_info(t=self._t)
        upper_x_lim, lower_x_limit = self.state_limits
        done = self._t >= self.max_episode_steps

        reward = torch.tensor(5., device=self.device)
        x_front_dim = [Platoon.X_FRONT + sum(self.n_dims[:i]) for i in range(self.n_systems)]
        x_back_dim = [Platoon.X_BACK + sum(self.n_dims[:i]) for i in range(self.n_systems)]
        state_x = self.state[x_front_dim + x_back_dim]
        reward -= torch.norm(state_x - self.params['r'])
        reward = np.clip(reward.cpu().numpy(), self.reward_limits[1], self.reward_limits[0])

        return self.state, float(reward), done, {}

    def render(self) -> np.ndarray:
        h = 1000
        w = 400
        fig, ax = plt.subplots(figsize=(h / 100, w / 100), dpi=100)
        canvas = FigureCanvas(fig)

        x_front_dim = [Platoon.X_FRONT + sum(self.n_dims[:i]) for i in range(self.n_systems)]
        x_front = self.state[x_front_dim]
        x_vehicle = np.zeros(self.n_systems)

        x_ref = self._info['x_ref']
        x_cur = copy.deepcopy(x_ref)
        for i, distance in enumerate(x_front):
            x_vehicle[i] = x_cur - distance
            x_cur -= distance

        length = 0.15
        width = 0.07
        plot_rectangle(ax, torch.tensor([x_ref, 0.]), 0., length, width, 'green')
        plot_rectangle(ax, torch.tensor([x_ref - self.params['r'] * (self.n_systems + 1), 0.]),
                       0., length, width, 'blue')

        for i_vehicle in range(self.n_systems):
            plot_rectangle(ax, torch.tensor([x_vehicle[i_vehicle], 0.]), 0., length, width, 'red')
            ax.text(x_vehicle[i_vehicle], 0.05, i_vehicle)

        x_ref = x_ref.cpu().detach().numpy()
        plt.xlim((x_ref - self.params['r'] * (self.n_systems + 2), x_ref + self.params['r']))
        plt.ylim((-0.1, 0.1))

        # get rgb array
        ax.axis('equal')
        canvas.draw()
        data = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()

        return data

    def plot_states(self, states: torch.Tensor) -> np.ndarray:
        # setup canvas
        fig, axs = plt.subplots(2, figsize=(5, 5), dpi=500)
        canvas = FigureCanvas(fig)

        # extract states
        x_front_dim = [Platoon.X_FRONT + sum(self.n_dims[:i]) for i in range(self.n_systems)]
        x_front = states[:, x_front_dim].cpu().detach().numpy()
        x_back_dim = [Platoon.X_BACK + sum(self.n_dims[:i]) for i in range(self.n_systems)]
        x_back = states[:, x_back_dim].cpu().detach().numpy()
        v_dim = [Platoon.V + sum(self.n_dims[:i]) for i in range(self.n_systems)]
        v = states[:, v_dim].cpu().detach().numpy()
        time = np.asarray([i * self.dt for i in range(x_front.shape[0])])
        total_time = np.arange(0, self.max_episode_steps * self.dt, self.dt)

        # plot x_front - x_back
        axs[0].plot(total_time, np.zeros_like(total_time), linestyle="-.")
        for i in range(x_front.shape[1]):
            axs[0].plot(time, x_front[:, i] - x_back[:, i], label=rf"$(\Delta x_f-\Delta x_b)_{i}$")
        axs[0].set_title(r'$(\Delta x_f-\Delta x_b)$')
        axs[0].set_xlabel('time (sec.)')
        axs[0].set_ylabel(r'$(\Delta x_f-\Delta x_b)$ (m)')
        if x_front.shape[1] <= 6:
            axs[0].legend(loc='upper right', bbox_to_anchor=(1.2, 1.05))
        axs[0].set_xlim((0., self.max_episode_steps * self.dt))

        # plot v
        axs[1].plot(total_time, self._ref_path[:, 1].cpu().detach().numpy(), linestyle="-.")
        for i in range(v.shape[1]):
            axs[1].plot(time, v[:, i], label=rf"$v_{i}$")
        axs[1].set_title(r'$v$')
        axs[1].set_xlabel('time (sec.)')
        axs[1].set_ylabel(r'$v$ (m/s)')
        if v.shape[1] <= 6:
            axs[1].legend(loc='upper right', bbox_to_anchor=(1.2, 1.05))
        axs[1].set_xlim((0., self.max_episode_steps * self.dt))

        # generate numpy data
        plt.tight_layout()
        canvas.draw()
        data = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()

        return data

    def default_param(self) -> dict:
        params = {
            'n': 5,
            'm': torch.tensor([1., 1., 1., 1., 1.], device=self.device),
            'v_init': 2.0,
            'r': 1.0,
            'speed profile': 'constant'
        }
        return params

    def validate_params(self, params: dict) -> bool:
        # check keys
        keys = params.keys()
        valid = 'n' in keys and 'm' in keys and 'v_init' in keys and 'r' in keys and 'speed profile' in keys

        # check dim
        dim = params['n']
        valid = valid and params['m'].ndim == 1 and params['m'].shape[0] == dim

        # check mass
        valid = valid and torch.count_nonzero(params['m']) == dim

        # check speed profile
        valid = valid and params['speed profile'] in Platoon.VALID_SPEED_PROFILE

        return valid

    def goal_mask(self, x: torch.Tensor) -> tuple:
        system_x = torch.split(x, list(self.n_dims), dim=1)
        masks = []
        for i_system in range(self.n_systems):
            mask = torch.ones(system_x[i_system].shape[0]).type_as(x)
            mask = torch.logical_and(
                mask, system_x[i_system][:, Platoon.X_FRONT] <= system_x[i_system][:, Platoon.X_BACK] + 0.02)
            mask = torch.logical_and(
                mask, system_x[i_system][:, Platoon.X_FRONT] >= system_x[i_system][:, Platoon.X_BACK] - 0.02)
            masks.append(mask.bool())
        return tuple(masks)

    def sample_goal(self, batch_size: int) -> torch.Tensor:
        states = self.sample_states(batch_size)
        x_front_dim = [Platoon.X_FRONT + sum(self.n_dims[:i]) for i in range(self.n_systems)]
        x_back_dim = [Platoon.X_BACK + sum(self.n_dims[:i]) for i in range(self.n_systems)]
        states[:, x_back_dim] = states[:, x_front_dim]
        return states

    @property
    def share_lyapunov(self) -> tuple:
        return (
            torch.tensor([0, self.n_systems - 1], device=self.device),
            torch.tensor([i + 1 for i in range(self.n_systems - 2)], device=self.device)
        )

    @property
    def share_ctrl(self) -> tuple:
        return (
            torch.tensor([0, self.n_systems - 1], device=self.device),
            torch.tensor([i + 1 for i in range(self.n_systems - 2)], device=self.device)
        )

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
        return Platoon.MAX_EPISODE_STEPS

    @property
    def state_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        upper_limit = torch.tensor(
            [self.params['r'] * 2, self.params['r'] * 2, self.params['v_init'] * 2], device=self.device
        ).repeat(self.n_systems)
        lower_limit = torch.tensor([0., 0., 0.], device=self.device).repeat(self.n_systems)

        return upper_limit, lower_limit

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
        goal = torch.tensor([self.params['r'], self.params['r'], self.params['v_init']], device=self.device)
        return goal.repeat(self.n_systems)

    def _u_vehicle(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Decentralized LQR controller

        Parameters
        ----------
        x: torch.Tensor
            decentralized states: x, v
        target: torch.Tensor
            target states: x, v

        Returns
        -------
        u: torch.Tensor
            control input for the single vehicle
        """
        if x.ndim == 1:
            x = x.unsqueeze(0)
        # goal = torch.zeros_like(x)
        goal = target
        A = np.eye(x.shape[1])
        A[0, 1] = 1. * self.dt
        B = np.zeros((x.shape[1], self.n_controls[0]))
        B[1, 0] = 1 / self.params['m'][0] * self.dt
        Q = np.eye(x.shape[1])
        # Q[1, 1] = 0.
        R = np.eye(self.n_controls[0])
        K = torch.tensor(lqr(A, B, Q, R), device=self.device, dtype=torch.float)
        return -(K @ (x - goal).T).T

    def u_nominal(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        x_car = torch.zeros(x.shape[0], device=self.device)
        target_v = torch.tensor(self.params['v_init'], device=self.device).unsqueeze(0).repeat(x.shape[0])
        states = []
        targets = []
        ctrls = []

        # extract states
        x_front_dim = [Platoon.X_FRONT + sum(self.n_dims[:i]) for i in range(self.n_systems)]
        x_front = x[:, x_front_dim]
        x_back_dim = [Platoon.X_BACK + sum(self.n_dims[:i]) for i in range(self.n_systems)]
        x_back = x[:, x_back_dim]
        v_dim = [Platoon.V + sum(self.n_dims[:i]) for i in range(self.n_systems)]
        v = x[:, v_dim]

        for i in range(self.n_systems):
            target_x = x_car - (x_front[:, i] + x_back[:, i]) / 2
            x_car -= x_front[:, i]
            states.append(torch.cat([x_car.unsqueeze(1), v[:, i].unsqueeze(1)], dim=1))
            targets.append(torch.cat([target_x.unsqueeze(1), target_v.unsqueeze(1)], dim=1))
            ctrls.append(self._u_vehicle(states[-1], targets[-1]))
        ctrl = torch.cat(ctrls, dim=1)
        return ctrl + self.u_eq

    @property
    def use_linearized_controller(self) -> bool:
        return False
