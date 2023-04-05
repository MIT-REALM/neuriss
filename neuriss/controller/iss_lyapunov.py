import torch
import torch.nn as nn
import os

from .controller import NeuralController
from neuriss.network.pd_quadratic import PDQuadraticNet
from neuriss.network.linear import ChiFunctions
from neuriss.env.base import Env


class NeuralISSLyapunov:
    """
    ISS-Lyapunov function parameterized by neural networks.

    Parameters
    ----------
    state_dims: tuple,
        dimension of states of each subsystem
    ctrl_dims: tuple,
        dimension of control inputs of each subsystem
    share_lyapunov: tuple,
        a tuple of Tensors, where each Tensor indicates the ids of the subsystem
        that share this iss-Lyapunov function
    share_ctrl: tuple,
        a tuple of Tensors, where each Tensor indicates the ids of the subsystem
        that share this controller
    device: torch.device,
        cpu or cuda
    goal_point: torch.Tensor,
        goal point of the whole system
    state_std: torch.Tensor,
        standard deviation of states, used for normalization
    u_eq: torch.Tensor,
        control input to stabilize the system at the equilibrium
    gamma: int = 0.01,
        constant gamma needed in the definition of ISS-Lyapunov functions
    lambd: int = 1.0,
        convergence rate
    """

    def __init__(
            self,
            state_dims: tuple,
            ctrl_dims: tuple,
            share_lyapunov: tuple,
            share_ctrl: tuple,
            device: torch.device,
            goal_point: torch.Tensor,
            state_std: torch.Tensor,
            u_eq: torch.Tensor,
            env: Env,
            gamma: int = 0.01,
            lambd: int = 1.0,
            residue: bool = False
    ):

        self.n_systems = len(state_dims)
        self.state_dims = state_dims
        self.ctrl_dims = ctrl_dims
        self.share_lyapunov = share_lyapunov
        self.n_lyapunov = len(share_lyapunov)
        self.share_ctrl = share_ctrl
        self.n_controller = len(share_ctrl)
        self.gammas = [gamma for _ in range(self.n_systems)]
        self.lambdas = [lambd for _ in range(self.n_systems)]
        self.device = device
        self.goal_point = goal_point.to(device)
        self.state_std = state_std.to(device)
        self.u_eq = u_eq.to(device)
        self.env = env  # needed for solving QP
        self.residue = residue

        # input dims of Lyapunov functions and gains
        self.input_dims_lya = torch.zeros(self.n_lyapunov, dtype=torch.int, device=device)

        # Lyapunov functions and gains for each subsystem
        self.lyapunov_funcs = []
        for i in range(self.n_lyapunov):
            # check the dims of the states are the same
            state_dims_torch = torch.tensor(self.state_dims, device=device)
            assert len(set(state_dims_torch[self.share_lyapunov[i]].cpu().numpy())) == 1
            self.input_dims_lya[i] = int(state_dims_torch[self.share_lyapunov[i]][0])

            # init Lyapunov function
            self.lyapunov_funcs.append(
                PDQuadraticNet(
                    in_dim=int(self.input_dims_lya[i]),
                    hidden_layers=(64, 64),
                    hidden_activation=nn.Tanh()
                ).to(self.device)
            )

        # init gain function
        self.chis = ChiFunctions(n=self.n_systems)

        # input dims of controllers
        self.input_dims_ctrl = torch.zeros(self.n_controller, dtype=torch.int, device=device)
        self.output_dims_ctrl = torch.zeros(self.n_controller, dtype=torch.int, device=device)

        # set up controllers
        self.controllers = []
        for i in range(self.n_controller):
            # check the dims of the states and ctrls are the same
            state_dims_torch = torch.tensor(self.state_dims, device=device)
            assert len(set(state_dims_torch[self.share_ctrl[i]].cpu().numpy())) == 1
            self.input_dims_ctrl[i] = int(state_dims_torch[self.share_ctrl[i]][0])
            ctrl_dims_torch = torch.tensor(self.ctrl_dims, device=device)
            assert len(set(ctrl_dims_torch[self.share_ctrl[i]].cpu().numpy())) == 1
            self.output_dims_ctrl[i] = int(ctrl_dims_torch[self.share_ctrl[i]][0])

            # init controllers
            self.controllers.append(
                NeuralController(
                    state_dim=int(self.input_dims_ctrl[i]),
                    action_dim=int(self.output_dims_ctrl[i]),
                    hidden_layers=(64, 64)
                ).to(self.device)
            )

        # establish the mapping from subsystem idx to function idx
        self._mapping_lya = torch.zeros(self.n_systems, dtype=torch.int, device=device)
        for i, share_id in enumerate(self.share_lyapunov):
            self._mapping_lya[share_id] = i
        self._mapping_ctrl = torch.zeros(self.n_systems, dtype=torch.int, device=device)
        for i, share_id in enumerate(self.share_ctrl):
            self._mapping_ctrl[share_id] = i

    def mapping_lya(self, idx: int) -> int:
        """
        Map the idx of subsystem to the idx of the Lyapunov functions.

        Parameters
        ----------
        idx: int,
            index of the subsystem

        Returns
        -------
        func_idx: int,
            index of the Lyapunov function
        """
        return self._mapping_lya[idx]

    def mapping_ctrl(self, idx: int) -> int:
        """
        Map the idx of subsystem to the idx of the controllers.

        Parameters
        ----------
        idx: int,
            index of the subsystem

        Returns
        -------
        func_idx: int,
            index of the controller
        """
        return self._mapping_ctrl[idx]

    def normalize_state(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize the states to be: mean 0, std 1.

        Parameters
        ----------
        x: torch.Tensor,
            input states

        Returns
        -------
        x_new: torch.Tensor,
            normalized states
        """
        return (x.to(self.device) - self.goal_point) / self.state_std

    def lyapunov_values(self, x: torch.Tensor) -> tuple:
        """
        Get the values of the ISS Lyapunov functions for each subsystem.

        Parameters
        ----------
        x: torch.Tensor,
            states

        Returns
        -------
        Vs: tuple,
            the values of the ISS Lyapunov functions of each subsystem
        """
        Vs = []
        system_x = torch.split(self.normalize_state(x), list(self.state_dims), dim=1)
        for i in range(self.n_systems):
            Vs.append(self.lyapunov_funcs[self.mapping_lya(i)](system_x[i]))
        return tuple(Vs)

    def lyapunov_with_jacobian(self, x: torch.Tensor) -> tuple:
        """
        Get the values of the ISS Lyapunov functions and its Jacobian for each subsystem.

        Parameters
        ----------
        x: torch.Tensor,
            states

        Returns
        -------
        Vs: tuple,
            the values of the ISS Lyapunov functions of each subsystem
        Js: tuple,
            the Jacobian of the ISS Lyapunov functions of each subsystem
        """
        Vs = []
        Js = []
        system_x = torch.split(self.normalize_state(x), list(self.state_dims), dim=1)
        for i in range(self.n_systems):
            V, J = self.lyapunov_funcs[self.mapping_lya(i)].forward_jacobian(system_x[i])
            Vs.append(V)
            Js.append(J)
        return tuple(Vs), tuple(Js)

    def gains(self, x: torch.Tensor, i: int) -> torch.Tensor:
        """
        Return max_j{chi_i(V_j(x_j)).

        Parameters
        ----------
        x: torch.Tensor,
            states
        i: int,
            the index of the subsystem

        Returns
        -------
        gain: torch.Tensor,
            max_j{chi_i(V_j(x_j))
        """
        Vs = self.lyapunov_values(x)
        gains = []
        for j in range(self.n_systems):
            if j != i:
                gains.append(self.chis.value(i, Vs[j]))
        system_x = torch.split(self.normalize_state(x), list(self.state_dims), dim=1)
        actions = self.controllers[self.mapping_ctrl(i)].forward(system_x[i])
        gains.append(torch.norm(actions, dim=1).unsqueeze(1) * self.gammas[i])
        return torch.max(torch.cat(gains, dim=1), dim=1)[0].unsqueeze(1)

    def u(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get control inputs.

        Parameters
        ----------
        x: torch.Tensor,
            states

        Returns
        -------
        u: torch.Tensor,
            control inputs
        """
        us = []
        system_x = torch.split(self.normalize_state(x), list(self.state_dims), dim=1)
        for i in range(self.n_systems):
            u = self.controllers[self.mapping_ctrl(i)].forward(system_x[i])
            us.append(u)
        if self.residue:
            return torch.cat(us, dim=1) + self.env.u_nominal(x)
        else:
            return torch.cat(us, dim=1) + self.u_eq

    def act(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get control inputs without gradients.

        Parameters
        ----------
        x: torch.Tensor,
            states

        Returns
        -------
        u: torch.Tensor,
            control inputs
        """
        us = []
        system_x = torch.split(self.normalize_state(x.unsqueeze(0)), list(self.state_dims), dim=1)
        for i in range(self.n_systems):
            with torch.no_grad():
                u = self.controllers[self.mapping_ctrl(i)].forward(system_x[i])
            us.append(u)
        if self.residue:
            return torch.cat(us, dim=1) + self.env.u_nominal(x)
        else:
            return torch.cat(us, dim=1) + self.u_eq

    def save(self, path: str):
        """
        Save the model.

        Parameters
        ----------
        path: str,
            path to save the model
        """
        for i_func in range(self.n_lyapunov):
            torch.save(self.lyapunov_funcs[i_func].state_dict(),
                       os.path.join(path, f'lyapunov_{i_func}.pkl'))
        torch.save(self.chis.state_dict(), os.path.join(path, f'chi.pkl'))
        for i_ctrl in range(self.n_controller):
            torch.save(self.controllers[i_ctrl].state_dict(),
                       os.path.join(path, f'controller_{i_ctrl}.pkl'))

    def load(self, path: str, device: torch.device, ctrl_only: bool = False):
        """
        Load the pre-trained model.

        Parameters
        ----------
        path: str,
            path to load the model
        device: torch.device,
            device to be used
        ctrl_only: bool,
            if true, only load the controller model
        """
        if not ctrl_only:
            for i_func in range(self.n_lyapunov):
                self.lyapunov_funcs[i_func].load_state_dict(
                    torch.load(os.path.join(path, f'lyapunov_{i_func}.pkl'), map_location=device))
        for i_ctrl in range(self.n_controller):
            self.controllers[i_ctrl].load_state_dict(
                torch.load(os.path.join(path, f'controller_{i_ctrl}.pkl'), map_location=device))
