import torch
import torch.nn as nn

from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from .ppo import PPO
from ..network.pd_quadratic import PDQuadraticNet
from ..env.base import Env


class LYPPO(PPO):
    """
    PPO with Lyapunov critics.

    References
    ----------
    [1] Ya-Chien Chang and Sicun Gao.
    Stabilizing neural control using self-learned almost lyapunov critics.
    In 2021 IEEE International Conference on Robotics and Automation (ICRA), pages 1803–1809. IEEE, 2021.
    """

    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            device: torch.device,
            goal_point: torch.Tensor,
            u_eq: torch.Tensor,
            state_std: torch.Tensor,
            ctrl_std: torch.Tensor,
            env: Env,
            gamma: float = 0.995,
            rollout_length: int = 4096,
            mix_buffer: int = 1,
            lr_actor: float = 3e-4,
            lr_critic: float = 3e-4,
            units_actor: tuple = (128, 128),
            units_critic: tuple = (128, 128),
            epoch_ppo: int = 20,
            clip_eps: float = 0.2,
            lambd: float = 0.97,
            coef_ent: float = 0.0,
            max_grad_norm: float = 10.0
    ):
        super(LYPPO, self).__init__(
            state_dim, action_dim, device, goal_point, u_eq, state_std, ctrl_std, gamma, rollout_length,
            mix_buffer, lr_actor, lr_critic, units_actor, units_critic, epoch_ppo, clip_eps, lambd,
            coef_ent, max_grad_norm
        )

        # change the critic to Lyapunov critic
        self.critic = PDQuadraticNet(
            in_dim=state_dim,
            hidden_layers=(64, 64),
            hidden_activation=nn.Tanh()
        ).to(device)
        self.optim_critic = Adam(self.critic.parameters(), lr=lr_critic)

        self.goal_point = goal_point
        self.state_std = state_std
        self.env = env

    def normalize_state(self, x: torch.Tensor) -> torch.Tensor:
        return (x.to(self.device) - self.goal_point) / self.state_std

    def fit_lyapunov(self) -> dict:
        eps = 1.0
        loss_record = {}
        for e in range(60):
            states, actions, rewards, dones, log_pis, next_states = self.buffer.sample(batch_size=128)

            # fit Lyapunov function
            update_step = 0
            while True:
                # calculate Lyapunov risk
                states_trans = self.normalize_state(states)
                V, JV = self.critic.forward_jacobian(states_trans)
                x_dot = self.env.closed_loop_dynamics(states, actions)
                V_dot = torch.bmm(JV, x_dot.unsqueeze(-1)).squeeze(1)
                V_goal = self.critic(self.env.sample_goal(batch_size=states.shape[0] // 5).to(self.device))

                # record the loss
                loss_record['decrease'] = (torch.relu(V_dot + eps)).mean()
                loss_record['goal'] = V_goal.pow(2).mean()
                loss_record['lyapunov_risk'] = loss_record['decrease'] + 10. * loss_record['goal']

                # update
                self.optim_critic.zero_grad()
                loss_record['lyapunov_risk'].backward()
                self.optim_critic.step()
                update_step += 1

                if update_step > 10:
                    break

        return loss_record

    def update(self, writer: SummaryWriter = None):
        self.learning_steps += 1
        loss_lyapunov = self.fit_lyapunov()
        states, actions, rewards, dones, log_pis, next_states = \
            self.buffer.get()
        self.update_ppo(states, actions, rewards, dones, log_pis, next_states)
        if writer is not None:
            for loss_name in loss_lyapunov.keys():
                writer.add_scalar(f'loss/{loss_name}', loss_lyapunov[loss_name].item(), self.learning_steps)

    def update_critic(self, states: torch.Tensor, targets: torch.Tensor):
        states_trans = self.normalize_state(states)
        loss_critic = (self.critic(states_trans) - targets).pow_(2).mean()

        self.optim_critic.zero_grad()
        loss_critic.backward(retain_graph=False)
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.optim_critic.step()
