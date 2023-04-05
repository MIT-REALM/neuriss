import torch
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from typing import Tuple, Optional
from copy import deepcopy

from .affine import ControlAffineEnv
from .utils import plot_rectangle, lqr


gravity = 9.81


class PlanarDrone(ControlAffineEnv):
    # state indices (same for each subsystem)
    XL = 0
    XR = 1
    YT = 2
    YB = 3
    THETA = 4
    VX = 5
    VY = 6
    OMEGA = 7

    # control indices (same for each subsystem)
    U1 = 0
    U2 = 1

    # max episode steps
    MAX_EPISODE_STEPS = 500

    # valid paths
    VALID_SPEED_PROFILE = (
        'constant',
        'static',
        'sin'
    )

    def __init__(
            self,
            device: torch.device,
            dt: float = 0.03,
            params: Optional[dict] = None
    ):
        super(PlanarDrone, self).__init__(device, dt, params)

        # set up parameters
        self.N_SYSTEMS = self.params['n_row'] * self.params['n_col']
        self.N_DIMS = tuple([8 for _ in range(self.N_SYSTEMS)])
        self.N_CONTROLS = tuple([2 for _ in range(self.N_SYSTEMS)])

        # tracking data
        self._ref_path = self._generate_ref(self.params['speed profile'])
        self._info = None

        # LQR controller
        self._K = None

    def __repr__(self):
        return 'PlanarDrone'

    def _set_info(self, t: int) -> dict:
        info = {}
        while t >= self._ref_path.shape[0]:
            t -= self._ref_path.shape[0]
        info['x'] = self._ref_path[t, 0]
        info['y'] = self._ref_path[t, 1]
        info['vx'] = self._ref_path[t, 2]
        info['vy'] = self._ref_path[t, 3]
        info['ax'] = self._ref_path[t, 4]
        info['ay'] = self._ref_path[t, 5]
        return info

    def _generate_ref(self, speed_profile: str) -> torch.Tensor:
        ref_path = torch.zeros(self.max_episode_steps, 6, device=self.device)

        if speed_profile == 'constant':
            x_ref = 0.0
            y_ref = 0.0
            vx_ref = self.params['vx_init']
            vy_ref = self.params['vy_init']
            for step in range(self.max_episode_steps):
                x_ref += self.dt * vx_ref
                y_ref += self.dt * vy_ref
                ref_path[step, :].copy_(torch.tensor([x_ref, y_ref, vx_ref, vy_ref, 0., 0.]))
        elif speed_profile == 'static':
            ref_path = torch.zeros(self.max_episode_steps, 6, device=self.device)
        elif speed_profile == 'sin':
            x_ref = 0.0
            y_ref = 0.0
            vx_ref = self.params['vx_init']
            vy_ref = self.params['vy_init']
            for step in range(self.max_episode_steps):
                ax_ref = 0.5 * np.sin(step * self.dt) - 0.25
                vx_ref += self.dt * ax_ref
                vx_ref = np.clip(vx_ref, a_min=0.5, a_max=4.0)
                x_ref += self.dt * vx_ref
                ref_path[step, :].copy_(torch.tensor([x_ref, y_ref, vx_ref, vy_ref, 0., 0.]))
        else:
            raise NotImplementedError('DroneFormation2D error: Unknown path')

        return ref_path

    def _extract_states(self, x: torch.Tensor) -> tuple:
        xl_dim = [PlanarDrone.XL + sum(self.n_dims[:i]) for i in range(self.n_systems)]
        xr_dim = [PlanarDrone.XR + sum(self.n_dims[:i]) for i in range(self.n_systems)]
        yt_dim = [PlanarDrone.YT + sum(self.n_dims[:i]) for i in range(self.n_systems)]
        yb_dim = [PlanarDrone.YB + sum(self.n_dims[:i]) for i in range(self.n_systems)]
        theta_dim = [PlanarDrone.THETA + sum(self.n_dims[:i]) for i in range(self.n_systems)]
        vx_dim = [PlanarDrone.VX + sum(self.n_dims[:i]) for i in range(self.n_systems)]
        vy_dim = [PlanarDrone.VY + sum(self.n_dims[:i]) for i in range(self.n_systems)]
        omega_dim = [PlanarDrone.OMEGA + sum(self.n_dims[:i]) for i in range(self.n_systems)]
        if x.ndim == 1:
            return x[xl_dim], x[xr_dim], x[yt_dim], x[yb_dim], x[theta_dim], x[vx_dim], x[vy_dim], x[omega_dim]
        else:
            return x[:, xl_dim], x[:, xr_dim], x[:, yt_dim], x[:, yb_dim], \
                   x[:, theta_dim], x[:, vx_dim], x[:, vy_dim], x[:, omega_dim]

    @property
    def _state_dims(self) -> tuple:
        xl_dim = [PlanarDrone.XL + sum(self.n_dims[:i]) for i in range(self.n_systems)]
        xr_dim = [PlanarDrone.XR + sum(self.n_dims[:i]) for i in range(self.n_systems)]
        yt_dim = [PlanarDrone.YT + sum(self.n_dims[:i]) for i in range(self.n_systems)]
        yb_dim = [PlanarDrone.YB + sum(self.n_dims[:i]) for i in range(self.n_systems)]
        theta_dim = [PlanarDrone.THETA + sum(self.n_dims[:i]) for i in range(self.n_systems)]
        vx_dim = [PlanarDrone.VX + sum(self.n_dims[:i]) for i in range(self.n_systems)]
        vy_dim = [PlanarDrone.VY + sum(self.n_dims[:i]) for i in range(self.n_systems)]
        omega_dim = [PlanarDrone.OMEGA + sum(self.n_dims[:i]) for i in range(self.n_systems)]
        return xl_dim, xr_dim, yt_dim, yb_dim, theta_dim, vx_dim, vy_dim, omega_dim

    @property
    def _ctrl_dims(self) -> tuple:
        u1_dim = [PlanarDrone.U1 + sum(self.n_controls[:i]) for i in range(self.n_systems)]
        u2_dim = [PlanarDrone.U2 + sum(self.n_controls[:i]) for i in range(self.n_systems)]
        return u1_dim, u2_dim

    def _f(self, x: torch.Tensor) -> torch.Tensor:
        bs = x.shape[0]
        f = torch.zeros((bs, sum(self.n_dims), 1)).type_as(x)

        xl, xr, yt, yb, theta, vx, vy, omega = self._extract_states(x)
        xl_dim, xr_dim, yt_dim, yb_dim, theta_dim, vx_dim, vy_dim, omega_dim = self._state_dims

        info = self._set_info(self._t)
        if self.mode == 'test':
            vx_ref = info['vx']
            vy_ref = info['vy']
        else:
            if np.random.rand() < 0.5:
                vx_ref = info['vx'].max()
            else:
                vx_ref = info['vx'].min()
            if np.random.rand() < 0.5:
                vy_ref = info['vy'].max()
            else:
                vy_ref = info['vy'].min()

        # calculate f matrix
        for i in range(self.n_systems):
            if i % self.params['n_col'] == 0:
                f[:, xl_dim[i], 0] = vx[:, i] - vx_ref
            else:
                f[:, xl_dim[i], 0] = vx[:, i] - vx[:, i - 1]
            if i % self.params['n_col'] == self.params['n_col'] - 1:
                f[:, xr_dim[i], 0] = vx_ref - vx[:, i]
            else:
                f[:, xr_dim[i], 0] = vx[:, i + 1] - vx[:, i]
            if i >= self.n_systems - self.params['n_col']:
                f[:, yt_dim[i], 0] = vy_ref - vy[:, i]
            else:
                f[:, yt_dim[i], 0] = vy[:, i + self.params['n_col']] - vy[:, i]
            if i < self.params['n_col']:
                f[:, yb_dim[i], 0] = vy[:, i] - vy_ref
            else:
                f[:, yb_dim[i], 0] = vy[:, i] - vy[:, i - self.params['n_col']]
            f[:, theta_dim[i], 0] = omega[:, i]
            f[:, vy_dim[i], 0] = -gravity

        return f

    def _g(self, x: torch.Tensor) -> torch.Tensor:
        bs = x.shape[0]
        g = torch.zeros((bs, sum(self.n_dims), sum(self.n_controls))).type_as(x)

        xl, xr, yt, yb, theta, vx, vy, omega = self._extract_states(x)
        xl_dim, xr_dim, yt_dim, yb_dim, theta_dim, vx_dim, vy_dim, omega_dim = self._state_dims
        u1_dim, u2_dim = self._ctrl_dims

        for i in range(self.n_systems):
            g[:, vx_dim[i], u1_dim[i]] = -torch.sin(theta[:, i]) / self.params['m'][i]
            g[:, vx_dim[i], u2_dim[i]] = -torch.sin(theta[:, i]) / self.params['m'][i]
            g[:, vy_dim[i], u1_dim[i]] = torch.cos(theta[:, i]) / self.params['m'][i]
            g[:, vy_dim[i], u2_dim[i]] = torch.cos(theta[:, i]) / self.params['m'][i]
            g[:, omega_dim[i], u1_dim[i]] = self.params['l'][i] / self.params['I'][i]
            g[:, omega_dim[i], u2_dim[i]] = -self.params['l'][i] / self.params['I'][i]

        return g

    def reset(self) -> torch.Tensor:
        self._t = 0
        self._ref_path = self._generate_ref(self.params["speed profile"])
        self._info = self._set_info(t=0)

        # goal positions
        def random_r():
            return 0.9 * self.params['r'] + np.random.rand() * 0.2 * self.params['r']
        xs, ys = [], []
        x, y = random_r(), random_r() * 0.1
        for i in range(self.n_systems):
            xs.append(x)
            ys.append(y)
            if i % self.params['n_col'] == self.params['n_col'] - 1:
                x = random_r()
                y += random_r() * 0.1
            else:
                x += random_r()
                y += (random_r() - self.params['r']) * 0.1

        state = torch.zeros(1, sum(self.n_dims), device=self.device)
        xl_dim, xr_dim, yt_dim, yb_dim, theta_dim, vx_dim, vy_dim, omega_dim = self._state_dims
        for i in range(self.n_systems):
            if i % self.params['n_col'] == 0:
                state[0, xl_dim[i]] = xs[i]
            else:
                state[0, xl_dim[i]] = xs[i] - xs[i - 1]
            if i % self.params['n_col'] == self.params['n_col'] - 1:
                state[0, xr_dim[i]] = (self.params['n_col'] + 1) * self.params['r'] - xs[i]
            else:
                state[0, xr_dim[i]] = xs[i + 1] - xs[i]
            if i >= self.n_systems - self.params['n_col']:
                state[0, yt_dim[i]] = (self.params['n_row'] + 1) * self.params['r'] - ys[i]
            else:
                state[0, yt_dim[i]] = ys[i + self.params['n_col']] - ys[i]
            if i < self.params['n_col']:
                state[0, yb_dim[i]] = ys[i]
            else:
                state[0, yb_dim[i]] = ys[i] - ys[i - self.params['n_col']]
            state[0, vx_dim[i]] = (np.random.rand() - 0.5) * 0.3 + self.params['vx_init']
            state[0, vy_dim[i]] = (np.random.rand() - 0.5) * 0.3 + self.params['vy_init']
            state[0, theta_dim[i]] = (np.random.rand() - 0.5) * 0.1
            state[0, omega_dim[i]] = (np.random.rand() - 0.5) * 0.1

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
        done = self._t >= self.max_episode_steps
        reward = torch.tensor(5., device=self.device)
        xl, xr, yt, yb, theta, vx, vy, omega = self._extract_states(self.state)
        reward -= torch.norm(xr - xl)
        reward -= torch.norm(yt - yb)
        reward = np.clip(reward.cpu().detach().numpy(), self.reward_limits[1], self.reward_limits[0])

        return self.state, float(reward), done, {}

    def render(self) -> np.ndarray:
        h = 1000
        w = 1000
        fig, ax = plt.subplots(figsize=(h / 100, w / 100), dpi=100)
        canvas = FigureCanvas(fig)

        info = self._set_info(self._t)
        xs, ys = [], []
        x, y = info['x'], info['y']
        xl, xr, yt, yb, theta, vx, vy, omega = self._extract_states(self.state)
        for i in range(self.n_systems):
            if i % self.params['n_col'] == 0:
                x = xl[i] + info['x']
            else:
                x += xl[i]
            if i < self.params['n_col']:
                y = yb[i] + info['y']
            else:
                y = yb[i] + ys[i - self.params['n_col']]
            xs.append(deepcopy(x))
            ys.append(deepcopy(y))

        for i in range(self.n_systems):
            plot_rectangle(ax, torch.tensor([xs[i], ys[i]]), theta[i], 0.1, 0.02, color='blue')
            ax.text(xs[i], ys[i], f'{i}')

        plot_rectangle(ax, torch.tensor([
            info['x'] + self.params['r'] * (self.params['n_col'] + 1) / 2,
            info['y'] + self.params['r'] * (self.params['n_row'] + 1) / 2]), 0., 0.1, 0.02, color='green')

        # get rgb array
        ax.set_xlim(info['x'], info['x'] + (self.params['n_col']) + 1 * self.params['r'])
        ax.set_ylim(info['y'], info['y'] + (self.params['n_row']) + 1 * self.params['r'])
        canvas.draw()
        data = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(
            fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()

        return data

    def plot_states(self, states: torch.Tensor) -> np.ndarray:
        # setup canvas
        fig, axs = plt.subplots(4, figsize=(5, 5), dpi=500)
        canvas = FigureCanvas(fig)

        xl, xr, yt, yb, theta, vx, vy, omega = self._extract_states(states)
        time = np.asarray([i * self.dt for i in range(xl.shape[0])])
        total_time = np.arange(0, self.max_episode_steps * self.dt, self.dt)

        # plot xr - xl
        axs[0].plot(total_time, np.zeros_like(total_time), linestyle='-.')
        for i in range(self.n_systems):
            axs[0].plot(time, xr[:, i] - xl[:, i], label=rf'$(\Delta x_r - \Delta x_l)_{i}$')
        axs[0].set_title(r'$(\Delta x_r - \Delta x_l)$')
        axs[0].set_xlabel('time (sec.)')
        axs[0].set_ylabel(r'$(\Delta x_r - \Delta x_l)$ (m)')
        if self.n_systems <= 6:
            axs[0].legend(loc='upper right', bbox_to_anchor=(1.2, 1.05))
        axs[0].set_xlim((0., self.max_episode_steps * self.dt))

        # plot yt - yb
        axs[1].plot(total_time, np.zeros_like(total_time), linestyle='-.')
        for i in range(self.n_systems):
            axs[1].plot(time, yt[:, i] - yb[:, i], label=rf'$(\Delta y_t - \Delta y_b)_{i}$')
        axs[1].set_title(r'$(\Delta y_t - \Delta y_b)$')
        axs[1].set_xlabel('time (sec.)')
        axs[1].set_ylabel(r'$(\Delta y_t - \Delta y_b)$ (m)')
        if self.n_systems <= 6:
            axs[1].legend(loc='upper right', bbox_to_anchor=(1.2, 1.05))
        axs[1].set_xlim((0., self.max_episode_steps * self.dt))

        # plot vx
        axs[2].plot(total_time, np.zeros_like(total_time), linestyle='-.')
        for i in range(self.n_systems):
            axs[2].plot(time, vx[:, i], label=rf'$v_x{i}$')
        axs[2].set_title(r'$v_x$')
        axs[2].set_xlabel('time (sec.)')
        axs[2].set_ylabel(r'$v_x$ (m/s)')
        if self.n_systems <= 6:
            axs[2].legend(loc='upper right', bbox_to_anchor=(1.2, 1.05))
        axs[2].set_xlim((0., self.max_episode_steps * self.dt))

        # plot vy
        axs[3].plot(total_time, np.zeros_like(total_time), linestyle='-.')
        for i in range(self.n_systems):
            axs[3].plot(time, vy[:, i], label=rf'$v_y{i}$')
        axs[3].set_title(r'$v_y$')
        axs[3].set_xlabel('time (sec.)')
        axs[3].set_ylabel(r'$v_y$ (m/s)')
        if self.n_systems <= 6:
            axs[3].legend(loc='upper right', bbox_to_anchor=(1.2, 1.05))
        axs[3].set_xlim((0., self.max_episode_steps * self.dt))

        # generate numpy data
        plt.tight_layout()
        canvas.draw()
        data = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()

        return data

    def default_param(self) -> dict:
        params = {
            'n_col': 3,
            'n_row': 2,
            'm': torch.tensor([1., 1., 1., 1., 1., 1.], device=self.device),
            'I': torch.tensor([1., 1., 1., 1., 1., 1.], device=self.device),
            'l': torch.tensor([1., 1., 1., 1., 1., 1.], device=self.device) * 0.3,
            'r': 1.,
            'vx_init': 0.,
            'vy_init': 0.,
            'speed profile': 'static'
        }
        return params

    def validate_params(self, params: dict) -> bool:
        # check keys
        keys = params.keys()
        valid = 'n_col' in keys and 'n_row' in keys and 'm' in keys and 'I' in keys and 'l' in keys and \
                'r' in keys and 'vx_init' in keys and 'vy_init' in keys and 'speed profile' in keys

        # check dim
        dim = params['n_row'] * params['n_col']
        valid = valid and params['m'].ndim == 1 and params['m'].shape[0] == dim
        valid = valid and params['I'].ndim == 1 and params['I'].shape[0] == dim
        valid = valid and params['l'].ndim == 1 and params['l'].shape[0] == dim

        # check mass, moment of inertia, and length
        valid = valid and torch.count_nonzero(params['m']) == dim
        valid = valid and torch.count_nonzero(params['I']) == dim
        valid = valid and torch.count_nonzero(params['l']) == dim

        # check speed profile
        valid = valid and params['speed profile'] in PlanarDrone.VALID_SPEED_PROFILE

        return valid

    def goal_mask(self, x: torch.Tensor) -> tuple:
        system_x = torch.split(x, list(self.n_dims), dim=1)
        masks = []
        for i in range(self.n_systems):
            mask = torch.ones(system_x[i].shape[0]).type_as(x)
            mask = torch.logical_and(
                mask, system_x[i][:, PlanarDrone.XL] <= system_x[i][:, PlanarDrone.XR] + 0.02
            )
            mask = torch.logical_and(
                mask, system_x[i][:, PlanarDrone.XL] >= system_x[i][:, PlanarDrone.XR] - 0.02
            )
            mask = torch.logical_and(
                mask, system_x[i][:, PlanarDrone.YT] <= system_x[i][:, PlanarDrone.YB] + 0.02
            )
            mask = torch.logical_and(
                mask, system_x[i][:, PlanarDrone.YT] >= system_x[i][:, PlanarDrone.YB] - 0.02
            )
            masks.append(mask.bool())
        return tuple(masks)

    def sample_goal(self, batch_size: int) -> torch.Tensor:
        states = self.sample_states(batch_size)
        xl_dim, xr_dim, yt_dim, yb_dim, theta_dim, vx_dim, vy_dim, omega_dim = self._state_dims
        states[:, xr_dim] = states[:, xl_dim]
        states[:, yb_dim] = states[:, yt_dim]
        states[:, theta_dim] = 0.
        states[:, omega_dim] = 0.
        return states

    @property
    def share_lyapunov(self) -> tuple:
        return torch.tensor([i for i in range(self.n_systems)], device=self.device),

    @property
    def share_ctrl(self) -> tuple:
        return torch.tensor([i for i in range(self.n_systems)], device=self.device),

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
        return PlanarDrone.MAX_EPISODE_STEPS

    @property
    def state_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        upper_limit = torch.tensor(
            [self.params['r'] * 5, self.params['r'] * 5, self.params['r'] * 5., self.params['r'] * 5., torch.pi / 2,
             self.params['vx_init'] * 2 + 5, self.params['vy_init'] * 2 + 5, torch.pi / 2], device=self.device
        ).repeat(self.n_systems)
        lower_limit = torch.tensor(
            [0., 0., 0., 0., -torch.pi / 2,
             -self.params['vx_init'] * 2 - 5, -self.params['vy_init'] * 2 - 5, -torch.pi / 2], device=self.device
        ).repeat(self.n_systems)
        return upper_limit, lower_limit

    @property
    def control_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        upper_limit = torch.ones(sum(self.n_controls), device=self.device, dtype=torch.float) * \
                      self.params['m'].max() * 10.
        lower_limit = -1.0 * upper_limit

        return upper_limit, lower_limit

    @property
    def reward_limits(self) -> Tuple[float, float]:
        return self.n_systems * 1., 0.0

    @property
    def goal_point(self) -> torch.Tensor:
        goal = torch.tensor([self.params['r'], self.params['r'], self.params['r'], self.params['r'], 0.,
                             self.params['vx_init'], self.params['vy_init'], 0.], device=self.device)
        return goal.repeat(self.n_systems)

    def _u_drone(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Decentralized LQR controller

        Parameters
        ----------
        x: torch.Tensor
            decentralized states: x, y, theta, v_x, v_y, omega
        target: torch.Tensor
            target: x, y, vx, vy

        Returns
        -------
        u: torch.Tensor
            control input for the single drone
        """
        if x.ndim == 1:
            x = x.unsqueeze(0)
        goal = torch.zeros_like(x)
        goal[:, :2] = target[:, :2]
        goal[:, 3:5] = target[:, 2:]

        A = np.eye(x.shape[1])
        A[0, 3] = 1. * self.dt
        A[1, 4] = 1. * self.dt
        A[2, 5] = 1. * self.dt
        A[3, 2] = -gravity * self.dt
        B = np.zeros((x.shape[1], self.n_controls[0]))
        B[4, 0] = 1 / self.params['m'][0] * self.dt
        B[4, 1] = 1 / self.params['m'][0] * self.dt
        B[5, 0] = self.params['l'][0] / self.params['I'][0] * self.dt
        B[5, 1] = -self.params['l'][0] / self.params['I'][0] * self.dt
        Q = np.eye(x.shape[1])
        R = np.eye(self.n_controls[0])
        self._K = torch.tensor(lqr(A, B, Q, R), device=self.device, dtype=torch.float)

        return -(self._K @ (x - goal).T).T

    def u_nominal(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        xs, ys = [], []
        info = self._set_info(self._t)
        x_drone, y_drone = 0., 0.
        vx_ref = torch.tensor(self.params['vx_init'], device=self.device).unsqueeze(0).repeat(x.shape[0])
        vy_ref = torch.tensor(self.params['vy_init'], device=self.device).unsqueeze(0).repeat(x.shape[0])
        targets = []
        ctrls = []
        states_drone = []
        xl, xr, yt, yb, theta, vx, vy, omega = self._extract_states(x)
        for i in range(self.n_systems):
            if i % self.params['n_col'] == 0:
                x_drone = xl[:, i]
                target_x = (xl[:, i] + xr[:, i]) / 2
            else:
                target_x = x_drone + (xl[:, i] + xr[:, i]) / 2
                x_drone += xl[:, i]
            if i < self.params['n_col']:
                target_y = (yt[:, i] + yb[:, i]) / 2
                y_drone = yb[:, i]
            else:
                target_y = ys[i - self.params['n_col']] + (yt[:, i] + yb[:, i]) / 2
                y_drone = yb[:, i] + ys[i - self.params['n_col']]
            xs.append(deepcopy(x_drone))
            ys.append(deepcopy(y_drone))
            targets.append(torch.cat([
                target_x.unsqueeze(1),
                target_y.unsqueeze(1),
                vx_ref.unsqueeze(1),
                vy_ref.unsqueeze(1)], dim=1)
            )
            states_drone.append(torch.cat(
                [x_drone.unsqueeze(1), y_drone.unsqueeze(1), theta[:, i].unsqueeze(1), vx[:, i].unsqueeze(1),
                 vy[:, i].unsqueeze(1), omega[:, i].unsqueeze(1)], dim=1))
            ctrls.append(self._u_drone(states_drone[-1], targets[-1]))
        ctrl = torch.cat(ctrls, dim=1)
        return ctrl + self.u_eq

    @property
    def use_linearized_controller(self) -> bool:
        return False

    @property
    def u_eq(self):
        u = torch.zeros(sum(self.n_controls), device=self.device)
        u1_dim, u2_dim = self._ctrl_dims
        for i in range(self.n_systems):
            u[u1_dim[i]] = 0.5 * gravity * self.params['m'][i]
            u[u2_dim[i]] = 0.5 * gravity * self.params['m'][i]
        return u
